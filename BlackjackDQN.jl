using POMDPs
using POMDPTools: SparseCat, Deterministic
using QuickPOMDPs: QuickPOMDP
using Flux
using Statistics
using Random
using Plots

############################
# STATE
############################
struct BJState
    player_sum::Int
    dealer_upcard::Int
    usable_ace::Bool
    deck_counts::NTuple{10, Int}
    terminal::Bool
    is_blackjack::Bool      # true if player got natural blackjack (21 on first two cards)
    can_double::Bool        # true only on first action of a hand
    can_split::Bool         # true only if first two cards are a pair
    doubled::Bool           # true if player doubled down (used in reward)
    first_card::Int         # tracks first card for split detection (0 if not applicable)
end

const ACTIONS = [:hit, :stick, :double, :split]

############################
# SHOE MANAGEMENT
############################
const NUM_DECKS = 6
const SHUFFLE_THRESHOLD = 0.65

function fresh_shoe(num_decks::Int = NUM_DECKS)
    return ntuple(i -> i == 10 ? 16 * num_decks : 4 * num_decks, 10)
end

function should_shuffle(deck::NTuple{10, Int}, num_decks::Int = NUM_DECKS)
    total_remaining = sum(deck)
    total_cards = 52 * num_decks
    fraction_remaining = total_remaining / total_cards
    return fraction_remaining < (1.0 - SHUFFLE_THRESHOLD)
end

mutable struct Shoe
    counts::NTuple{10, Int}
    num_decks::Int
    running_count::Int
end

Shoe(num_decks::Int = NUM_DECKS) = Shoe(fresh_shoe(num_decks), num_decks, 0)

function maybe_shuffle!(shoe::Shoe)
    if should_shuffle(shoe.counts, shoe.num_decks)
        shoe.counts = fresh_shoe(shoe.num_decks)
        shoe.running_count = 0
        return true
    end
    return false
end

############################
# HI-LO COUNT
############################
function hilo_delta(card::Int)
    if card >= 2 && card <= 6
        return 1
    elseif card >= 7 && card <= 9
        return 0
    else
        return -1
    end
end

function true_count(shoe::Shoe)
    total_remaining = sum(shoe.counts)
    decks_remaining = max(total_remaining / 52.0, 0.5)
    return shoe.running_count / decks_remaining
end

############################
# HELPERS
############################
function draw_card_dist(deck_counts)
    total = sum(deck_counts)
    probs = [c / total for c in deck_counts]
    return SparseCat(1:10, probs)
end

function remove_card(deck, card)
    return ntuple(i -> i == card ? deck[i] - 1 : deck[i], 10)
end

function update_sum(sum_val, usable_ace, card)
    if card == 1
        if sum_val + 11 <= 21
            return sum_val + 11, true
        else
            return sum_val + 1, usable_ace
        end
    else
        return sum_val + card, usable_ace
    end
end

function adjust_for_ace(sum_val, usable_ace)
    if sum_val > 21 && usable_ace
        return sum_val - 10, false
    else
        return sum_val, usable_ace
    end
end

# Card value for split detection (face cards all count as 10)
function card_value_for_split(card::Int)
    return card  # already normalized: 1=Ace, 2-9, 10=all ten-value cards
end

############################
# DEALER
############################
function dealer_outcomes(dealer_upcard, deck)
    outcomes = Dict{Int, Float64}()

    function recurse(dealer_sum, usable, deck, prob)
        dealer_sum, usable = adjust_for_ace(dealer_sum, usable)

        if dealer_sum >= 17
            outcomes[dealer_sum] = get(outcomes, dealer_sum, 0.0) + prob
            return
        end

        total = sum(deck)
        for card in 1:10
            if deck[card] == 0
                continue
            end
            p = deck[card] / total
            new_deck = remove_card(deck, card)
            new_sum, new_usable = update_sum(dealer_sum, usable, card)
            recurse(new_sum, new_usable, new_deck, prob * p)
        end
    end

    init_sum = dealer_upcard == 1 ? 11 : dealer_upcard
    init_usable = dealer_upcard == 1 ? true : false
    recurse(init_sum, init_usable, deck, 1.0)
    return outcomes
end

############################
# TRANSITION
############################
function bj_transition(s::BJState, a)
    if s.terminal
        return Deterministic(s)
    end

    deck = s.deck_counts

    # --- HIT ---
    if a == :hit
        dist = draw_card_dist(deck)
        states = BJState[]
        probs  = Float64[]

        for card in 1:10
            p = pdf(dist, card)
            if p == 0.0; continue; end

            new_deck = remove_card(deck, card)
            new_sum, usable = update_sum(s.player_sum, s.usable_ace, card)
            new_sum, usable = adjust_for_ace(new_sum, usable)
            terminal = new_sum > 21

            push!(states, BJState(
                new_sum, s.dealer_upcard, usable, new_deck,
                terminal, false, false, false, false, 0
            ))
            push!(probs, p)
        end
        return SparseCat(states, probs)
    end

    # --- STICK ---
    if a == :stick
        dealer_out = dealer_outcomes(s.dealer_upcard, s.deck_counts)
        states = BJState[]
        probs  = Float64[]

        for (dealer_sum, p) in dealer_out
            push!(states, BJState(
                s.player_sum, dealer_sum, s.usable_ace, s.deck_counts,
                true, s.is_blackjack, false, false, s.doubled, 0
            ))
            push!(probs, p)
        end
        return SparseCat(states, probs)
    end

    # --- DOUBLE ---
    # Player draws exactly one more card, then hand ends (forced stick)
    if a == :double && s.can_double
        dist = draw_card_dist(deck)
        states = BJState[]
        probs  = Float64[]

        for card in 1:10
            p = pdf(dist, card)
            if p == 0.0; continue; end

            new_deck = remove_card(deck, card)
            new_sum, usable = update_sum(s.player_sum, s.usable_ace, card)
            new_sum, usable = adjust_for_ace(new_sum, usable)

            if new_sum > 21
                # Bust immediately after double — terminal
                push!(states, BJState(
                    new_sum, s.dealer_upcard, usable, new_deck,
                    true, false, false, false, true, 0
                ))
                push!(probs, p)
            else
                # Must now play out dealer — resolve immediately
                dealer_out = dealer_outcomes(s.dealer_upcard, new_deck)
                for (dealer_sum, dp) in dealer_out
                    push!(states, BJState(
                        new_sum, dealer_sum, usable, new_deck,
                        true, false, false, false, true, 0
                    ))
                    push!(probs, p * dp)
                end
            end
        end
        return SparseCat(states, probs)
    end

    # --- SPLIT ---
    # Player splits the pair. Each hand starts fresh with one of the pair cards.
    # We model this as: start a new hand with the split card as the only card,
    # draw one more card to form the new hand, then play continues normally.
    if a == :split && s.can_split
        split_card = s.first_card
        dist = draw_card_dist(deck)
        states = BJState[]
        probs  = Float64[]

        for card in 1:10
            p = pdf(dist, card)
            if p == 0.0; continue; end

            new_deck = remove_card(deck, card)

            # New hand starts with split_card + drawn card
            new_sum, usable = update_sum(0, false, split_card)
            new_sum, usable = update_sum(new_sum, usable, card)
            new_sum, usable = adjust_for_ace(new_sum, usable)

            # Can double on new hand if sum allows, cannot re-split
            can_double_new = true
            can_split_new  = false  # simplification: no re-splitting

            terminal = new_sum > 21

            push!(states, BJState(
                new_sum, s.dealer_upcard, usable, new_deck,
                terminal, false, can_double_new, can_split_new, false, 0
            ))
            push!(probs, p)
        end
        return SparseCat(states, probs)
    end

    # Fallback: treat invalid action as stick
    return bj_transition(s, :stick)
end

############################
# REWARD
############################
function bj_reward(s::BJState, a, sp::BJState)
    if s.terminal
        return 0.0
    end
    if !sp.terminal
        return 0.0
    end

    # Determine bet multiplier (double = 2x stake)
    multiplier = sp.doubled ? 2.0 : 1.0

    # Player bust
    if sp.player_sum > 21
        return -1.0 * multiplier
    end

    dealer_sum = sp.dealer_upcard  # overloaded as final dealer result

    # Natural blackjack pays 3:2 (only on original non-doubled hand)
    if sp.is_blackjack && !sp.doubled
        return 1.5
    end

    if dealer_sum > 21 || sp.player_sum > dealer_sum
        return 1.0 * multiplier
    elseif sp.player_sum == dealer_sum
        return 0.0
    else
        return -1.0 * multiplier
    end
end

############################
# INITIAL STATE (SHOE-AWARE)
############################
function build_initial_dist(shoe::Shoe)
    maybe_shuffle!(shoe)

    init_deck = shoe.counts
    total_cards = sum(init_deck)

    states = BJState[]
    probs  = Float64[]

    # Deal dealer upcard, then two cards to player
    for dealer_card in 1:10
        p_dealer = init_deck[dealer_card] / total_cards
        if p_dealer == 0.0; continue; end

        deck_after_dealer = remove_card(init_deck, dealer_card)
        total_after_dealer = sum(deck_after_dealer)

        for card1 in 1:10
            p1 = deck_after_dealer[card1] / total_after_dealer
            if p1 == 0.0; continue; end

            deck_after_c1 = remove_card(deck_after_dealer, card1)
            total_after_c1 = sum(deck_after_c1)

            for card2 in 1:10
                p2 = deck_after_c1[card2] / total_after_c1
                if p2 == 0.0; continue; end

                deck_after_c2 = remove_card(deck_after_c1, card2)

                # Compute player sum
                sum1, usable1 = update_sum(0, false, card1)
                player_sum, usable_ace = update_sum(sum1, usable1, card2)
                player_sum, usable_ace = adjust_for_ace(player_sum, usable_ace)

                # Natural blackjack check
                is_bj = (player_sum == 21)

                # Can split if both cards are same value
                can_split = (card_value_for_split(card1) == card_value_for_split(card2))

                # Can double on any two-card hand
                can_double = true

                # Terminal immediately if natural blackjack
                terminal = is_bj

                push!(states, BJState(
                    player_sum, dealer_card, usable_ace, deck_after_c2,
                    terminal, is_bj, can_double, can_split, false, card1
                ))
                push!(probs, p_dealer * p1 * p2)
            end
        end
    end

    # Normalize in case of floating point drift
    total_prob = sum(probs)
    probs ./= total_prob

    return SparseCat(states, probs)
end

############################
# MDP
############################
m_sixdeck = QuickPOMDP(
    actions = ACTIONS,
    discount = 1.0,

    transition = bj_transition,
    reward = bj_reward,
    initialstate = build_initial_dist(Shoe(NUM_DECKS)),

    observation = (a, sp) -> Deterministic(sp),
    obstype = BJState,

    isterminal = s -> s.terminal
)

############################
# VALID ACTIONS
############################
# Returns only the actions legal in state s
function valid_actions(s::BJState)
    acts = [:hit, :stick]
    if s.can_double; push!(acts, :double); end
    if s.can_split;  push!(acts, :split);  end
    return acts
end

############################
# STATE VECTOR (HI-LO)
############################
function state_to_vec(s::BJState, tc::Float32)
    return Float32.([
        s.player_sum / 21.0,
        s.dealer_upcard / 10.0,
        s.usable_ace  ? 1.0 : 0.0,
        s.can_double  ? 1.0 : 0.0,
        s.can_split   ? 1.0 : 0.0,
        clamp(tc / 10.0, -1.0, 1.0)
    ])
end

const ACTION_TO_IDX = Dict(:hit => 1, :stick => 2, :double => 3, :split => 4)
const IDX_TO_ACTION = Dict(1 => :hit, 2 => :stick, 3 => :double, 4 => :split)

############################
# REPLAY BUFFER
############################
mutable struct ReplayBuffer
    capacity::Int
    memory::Vector{NamedTuple}
    pos::Int
end

ReplayBuffer(capacity::Int) = ReplayBuffer(capacity, NamedTuple[], 1)

function push_transition!(buffer::ReplayBuffer, transition)
    if length(buffer.memory) < buffer.capacity
        push!(buffer.memory, transition)
    else
        buffer.memory[buffer.pos] = transition
        buffer.pos = mod1(buffer.pos + 1, buffer.capacity)
    end
end

function sample_batch(buffer::ReplayBuffer, batch_size::Int)
    return rand(buffer.memory, min(batch_size, length(buffer.memory)))
end

############################
# NETWORK
############################
function create_q_network(input_size::Int, output_size::Int)
    return Chain(
        Dense(input_size => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 64,  relu),
        Dense(64  => output_size)
    )
end

############################
# DQN TRAINING
############################
function train_dqn!(mdp; n_episodes=100_000, batch_size=256, capacity=500_000, γ=1.0)
    input_size  = 6   # player_sum, dealer_upcard, usable_ace, can_double, can_split, true_count
    output_size = 4   # hit, stick, double, split

    q_net      = create_q_network(input_size, output_size)
    target_net = deepcopy(q_net)
    opt_state  = Flux.setup(Adam(5e-5), q_net)
    buffer     = ReplayBuffer(capacity)

    target_update_freq = 50  0
    global_step = 0
    returns     = Float64[]
    shoe        = Shoe(NUM_DECKS)

    best_net    = deepcopy(q_net)
    best_return = -Inf

    for ep in 1:n_episodes
        ϵ = max(0.01, 1.0 - ep / (n_episodes * 0.9))

        dist = build_initial_dist(shoe)
        s    = rand(dist)
        shoe.running_count += hilo_delta(s.dealer_upcard)

        ep_reward = 0.0

        while !s.terminal
            global_step += 1

            tc    = Float32(true_count(shoe))
            s_vec = state_to_vec(s, tc)

            # Mask invalid actions by setting their Q-values to -Inf
            legal = valid_actions(s)
            if rand() < ϵ
                a = rand(legal)
            else
                q_vals = q_net(s_vec)
                masked = fill(-Inf32, 4)
                for act in legal
                    masked[ACTION_TO_IDX[act]] = q_vals[ACTION_TO_IDX[act]]
                end
                a = IDX_TO_ACTION[argmax(masked)]
            end

            sp = rand(bj_transition(s, a))
            r  = bj_reward(s, a, sp)
            ep_reward += r

            # Update Hi-Lo count for drawn card
            if a == :hit || a == :double || a == :split
                for i in 1:10
                    if sp.deck_counts[i] < s.deck_counts[i]
                        shoe.running_count += hilo_delta(i)
                        break
                    end
                end
            end

            tc_sp  = Float32(true_count(shoe))
            sp_vec = state_to_vec(sp, tc_sp)

            push_transition!(buffer, (
                s    = s_vec,
                a    = ACTION_TO_IDX[a],
                r    = Float32(r),
                sp   = sp_vec,
                term = sp.terminal,
                legal_sp = Int[ACTION_TO_IDX[act] for act in valid_actions(sp)]
            ))

            if length(buffer.memory) >= batch_size
                batch = sample_batch(buffer, batch_size)

                S_batch    = hcat([b.s  for b in batch]...)
                Sp_batch   = hcat([b.sp for b in batch]...)
                R_batch    = Float32.([b.r    for b in batch])
                Term_batch = Bool.([b.term for b in batch])
                A_batch    = [b.a for b in batch]

                # Double DQN with action masking on next state
                main_next_q = q_net(Sp_batch)
                best_next_actions = map(1:length(batch)) do i
                    masked = fill(-Inf32, 4)
                    for idx in batch[i].legal_sp
                        masked[idx] = main_next_q[idx, i]
                    end
                    argmax(masked)
                end

                target_next_q = target_net(Sp_batch)
                max_target_q  = Float32.([
                    target_next_q[best_next_actions[i], i] for i in 1:length(batch)
                ])

                y = R_batch .+ Float32(γ) .* max_target_q .* .!Term_batch

                loss, grads = Flux.withgradient(q_net) do net
                    preds          = net(S_batch)
                    selected_preds = [preds[A_batch[i], i] for i in 1:length(batch)]
                    Flux.mse(selected_preds, y)
                end

                Flux.update!(opt_state, q_net, grads[1])
            end

            if global_step % target_update_freq == 0
                target_net = deepcopy(q_net)
            end

            s = sp
        end

        shoe.counts = s.deck_counts
        push!(returns, ep_reward)

        if ep % 100 == 0
            eval_scores = evaluate_dqn(mdp, q_net, n_episodes=1000)
            mean_return = mean(eval_scores)
            println("Episode $ep | ϵ: $(round(ϵ, digits=3)) | TC: $(round(true_count(shoe), digits=2)) | Train (last 500): $(round(mean(returns[max(1,end-499):end]), digits=3)) | Eval: $(round(mean_return, digits=3))")

            if mean_return > best_return
                best_return = mean_return
                best_net    = deepcopy(q_net)
                println("  ↑ New best network saved (return: $(round(best_return, digits=4)))")
            end
        end
    end

    println("\nBest network achieved mean return: $(round(best_return, digits=4))")
    return best_net, returns
end

############################
# EVALUATION
############################
function evaluate_dqn(mdp, trained_q_net; n_episodes=10_000)
    returns = Float64[]
    shoe    = Shoe(NUM_DECKS)

    for _ in 1:n_episodes
        dist = build_initial_dist(shoe)
        s    = rand(dist)
        shoe.running_count += hilo_delta(s.dealer_upcard)

        ep_r = 0.0
        while !s.terminal
            tc    = Float32(true_count(shoe))
            s_vec = state_to_vec(s, tc)

            q_vals = trained_q_net(s_vec)
            legal  = valid_actions(s)
            masked = fill(-Inf32, 4)
            for act in legal
                masked[ACTION_TO_IDX[act]] = q_vals[ACTION_TO_IDX[act]]
            end
            a = IDX_TO_ACTION[argmax(masked)]

            sp    = rand(bj_transition(s, a))
            ep_r += bj_reward(s, a, sp)

            if a == :hit || a == :double || a == :split
                for i in 1:10
                    if sp.deck_counts[i] < s.deck_counts[i]
                        shoe.running_count += hilo_delta(i)
                        break
                    end
                end
            end

            s = sp
        end

        shoe.counts = s.deck_counts
        push!(returns, ep_r)
    end

    return returns
end

############################
# BASELINE
############################
function baseline_policy(m, s)
    return s.player_sum >= 17 ? :stick : :hit
end

function evaluate_baseline(mdp; n_episodes=10_000)
    shoe    = Shoe(NUM_DECKS)
    returns = Float64[]

    for _ in 1:n_episodes
        dist = build_initial_dist(shoe)
        s    = rand(dist)
        ep_r = 0.0

        while !s.terminal
            a    = baseline_policy(mdp, s)
            sp   = rand(bj_transition(s, a))
            ep_r += bj_reward(s, a, sp)
            s    = sp
        end

        shoe.counts = s.deck_counts
        push!(returns, ep_r)
    end

    return returns
end

############################
# LEARNING CURVE
############################
function plot_learning_curve(returns; window=1000)
    n  = length(returns)
    xs = window:window:n
    ys = [mean(returns[i-window+1:i]) for i in xs]

    return plot(
        xs, ys,
        xlabel = "Episode",
        ylabel = "Avg Return (rolling $(window)-ep window)",
        label  = "DQN (6-deck, Hi-Lo)",
        legend = :bottomright,
        title  = "DQN Learning Curve"
    )
end

############################
# EXECUTION
############################
println("Evaluating baseline (stick on 17+)...")
baseline_scores = evaluate_baseline(m_sixdeck)
println("Baseline Mean Return: ", round(mean(baseline_scores), digits=4))
println("Baseline Std Error:   ", round(std(baseline_scores) / sqrt(length(baseline_scores)), digits=4))

println("\nTraining DQN on 6-deck shoe with Hi-Lo state...")
trained_model, train_history = train_dqn!(m_sixdeck, n_episodes=100_000)

println("\nEvaluating DQN...")
dqn_scores = evaluate_dqn(m_sixdeck, trained_model)

println("\n" * "="^40)
println("         6-DECK DQN RESULTS")
println("="^40)
println("Baseline (Stick on 17+):")
println("  Mean Return: ", round(mean(baseline_scores), digits=4))
println("  Std Error:   ", round(std(baseline_scores) / sqrt(length(baseline_scores)), digits=4))
println("-"^40)
println("DQN (Hi-Lo state, 6-deck shoe):")
println("  Mean Return: ", round(mean(dqn_scores), digits=4))
println("  Std Error:   ", round(std(dqn_scores) / sqrt(length(dqn_scores)), digits=4))
println("="^40)

p = plot_learning_curve(train_history, window=1000)
display(p)