using POMDPs
using POMDPTools: SparseCat, Deterministic
using QuickPOMDPs: QuickPOMDP
using CommonRLInterface: AbstractEnv
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
    deck_counts::NTuple{10, Int}  # [A,2,...,10]
    terminal::Bool
end

const ACTIONS = [:hit, :stick]

############################
# SHOE MANAGEMENT
############################

const NUM_DECKS = 1
const SHUFFLE_THRESHOLD = 0.65  # shuffle when 65% of cards remain

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
end

Shoe(num_decks::Int = NUM_DECKS) = Shoe(fresh_shoe(num_decks), num_decks)

function maybe_shuffle!(shoe::Shoe)
    if should_shuffle(shoe.counts, shoe.num_decks)
        shoe.counts = fresh_shoe(shoe.num_decks)
        return true
    end
    return false
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

    # HIT
    if a == :hit
        dist = draw_card_dist(deck)

        states = BJState[]
        probs = Float64[]

        for card in 1:10
            p = pdf(dist, card)
            if p == 0.0
                continue
            end

            new_deck = remove_card(deck, card)

            new_sum, usable = update_sum(s.player_sum, s.usable_ace, card)
            new_sum, usable = adjust_for_ace(new_sum, usable)

            terminal = new_sum > 21

            push!(states, BJState(
                new_sum,
                s.dealer_upcard,
                usable,
                new_deck,
                terminal
            ))
            push!(probs, p)
        end

        return SparseCat(states, probs)
    end

    # STICK
    if a == :stick
        dealer_out = dealer_outcomes(s.dealer_upcard, s.deck_counts)

        states = BJState[]
        probs = Float64[]

        for (dealer_sum, p) in dealer_out
            terminal_state = BJState(
                s.player_sum,
                dealer_sum,
                s.usable_ace,
                s.deck_counts,
                true
            )
            push!(states, terminal_state)
            push!(probs, p)
        end

        return SparseCat(states, probs)
    end

    return Deterministic(s)
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

    if sp.player_sum > 21
        return -1.0
    end

    dealer_sum = sp.dealer_upcard

    if dealer_sum > 21 || sp.player_sum > dealer_sum
        return 1.0
    elseif sp.player_sum == dealer_sum
        return 0.0
    else
        return -1.0
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
    probs = Float64[]

    for card in 1:10
        p = init_deck[card] / total_cards
        if p == 0.0
            continue
        end

        new_deck = remove_card(init_deck, card)

        push!(states, BJState(0, card, false, new_deck, false))
        push!(probs, p)
    end

    return SparseCat(states, probs)
end

############################
# MDP
############################
m_onedeck = QuickPOMDP(
    actions = ACTIONS,
    discount = 1.0,

    transition = bj_transition,
    reward = bj_reward,
    initialstate = build_initial_dist(Shoe()),  # single static initial dist

    observation = (a, sp) -> Deterministic(sp),
    obstype = BJState,

    isterminal = s -> s.terminal
)

############################
# POLICY
############################
function baseline_policy(m, s)
    if s.player_sum > 15
        return :stick
    else
        return :hit
    end
end

############################
# ROLLOUT (SHOE-AWARE)
############################
function rollout(mdp, policy_function, shoe::Shoe, max_steps::Int)
    r_total = 0.0
    disc = 1.0
    γ = discount(mdp)

    dist = build_initial_dist(shoe)
    s = rand(dist)

    for t in 1:max_steps
        if isterminal(mdp, s)
            shoe.counts = s.deck_counts
            break
        end

        a = policy_function(mdp, s)

        sp = rand(bj_transition(s, a))
        r = bj_reward(s, a, sp)

        r_total += disc * r
        disc *= γ
        s = sp
    end

    return r_total
end

############################
# SIMULATION LOOP
############################
function simulate(mdp, policy_function, num_episodes::Int, max_steps::Int = 50)
    shoe = Shoe(NUM_DECKS)
    total_reward = 0.0
    shuffle_count = 0

    for ep in 1:num_episodes
        cards_before = sum(shoe.counts)

        r = rollout(mdp, policy_function, shoe, max_steps)
        total_reward += r

        cards_after = sum(shoe.counts)
        if cards_after > cards_before
            shuffle_count += 1
        end
    end

    println("Episodes:    $num_episodes")
    println("Shuffles:    $shuffle_count")
    println("Mean reward: $(total_reward / num_episodes)")

    return total_reward / num_episodes
end

############################
# POLICY
############################

# Baseline 1: Always hit on 16 or below, stick on 17+
function baseline_hit17_policy(m, s)
    if s.player_sum >= 17
        return :stick
    else
        return :hit
    end
end

# Baseline 2: Always hit (never stick before 21)
function baseline_always_hit_policy(m, s)
    if s.player_sum >= 21
        return :stick
    else
        return :hit
    end
end

############################
# ROLLOUT (SHOE-AWARE)
############################
function rollout(mdp, policy_function, shoe::Shoe, max_steps::Int = 50)
    r_total = 0.0
    disc = 1.0
    γ = discount(mdp)

    dist = build_initial_dist(shoe)
    s = rand(dist)

    for t in 1:max_steps
        if isterminal(mdp, s)
            shoe.counts = s.deck_counts
            break
        end

        a = policy_function(mdp, s)
        sp = rand(bj_transition(s, a))
        r = bj_reward(s, a, sp)

        r_total += disc * r
        disc *= γ
        s = sp
    end

    return r_total
end

############################
# BASELINE SIMULATION
############################
function simulate_baseline(mdp, policy_function, num_episodes::Int)
    shoe = Shoe(NUM_DECKS)
    scores = Float64[]
    shuffle_count = 0

    for _ in 1:num_episodes
        cards_before = sum(shoe.counts)
        r = rollout(mdp, policy_function, shoe)
        cards_after = sum(shoe.counts)
        if cards_after > cards_before
            shuffle_count += 1
        end
        push!(scores, r)
    end

    println("Shuffles: $shuffle_count")
    return scores
end

println("Evaluating baseline: stick on 17...")
baseline_score = simulate_baseline(m_onedeck, baseline_hit17_policy, 10000)
println("Mean reward: ", mean(baseline_score))
println("Std error:   ", std(baseline_score) / sqrt(length(baseline_score)))

println("\nEvaluating baseline: always hit...")
baseline_always_hit_score = simulate_baseline(m_onedeck, baseline_always_hit_policy, 10000)
println("Mean reward: ", mean(baseline_always_hit_score))
println("Std error:   ", std(baseline_always_hit_score) / sqrt(length(baseline_always_hit_score)))

# ----------- COMPACT STATE KEY FOR Q-TABLE ----------- #
# Deck counts are used for transition probabilities but NOT for Q-table lookup.
# The Q-table is keyed on the decision-relevant state only.
function q_key(s::BJState)
    return (s.player_sum, s.dealer_upcard, s.usable_ace)
end

# ----------- Q-LEARNING ----------- #
function qlearning_episode!(Q, shoe::Shoe; ϵ=0.10, γ=1.0, α=0.2)
    function policy(s)
        if rand() < ϵ
            return rand(ACTIONS)
        else
            key = q_key(s)
            q_vals = [get(Q, (key, a), 0.0) for a in ACTIONS]
            return ACTIONS[argmax(q_vals)]
        end
    end

    dist = build_initial_dist(shoe)
    s = rand(dist)
    a = policy(s)

    steps = 0

    while !s.terminal
        # Transition still uses full BJState so deck probabilities are correct
        sp = rand(bj_transition(s, a))
        r = bj_reward(s, a, sp)

        # Q-table update uses compact key
        key    = q_key(s)
        key_sp = q_key(sp)

        best_next_q = maximum([get(Q, (key_sp, a_next), 0.0) for a_next in ACTIONS])
        current_q   = get(Q, (key, a), 0.0)
        Q[(key, a)] = current_q + α * (r + γ * best_next_q - current_q)

        s = sp
        a = policy(sp)
        steps += 1
    end

    shoe.counts = s.deck_counts
    return steps
end

function qlearning!(mdp; n_episodes=10000, α=0.2, eval_every=500)
    Q = Dict{Tuple{Tuple{Int,Int,Bool}, Symbol}, Float64}()
    shoe = Shoe(NUM_DECKS)

    curve_xs = Int[]
    curve_ys = Float64[]
    cumulative_steps = 0

    best_Q = copy(Q)
    best_return = -Inf

    for i in 1:n_episodes
        ϵ = max(0.1, 1 - 2 * i/(n_episodes))
        steps = qlearning_episode!(Q, shoe; ϵ=ϵ, α=α)
        cumulative_steps += steps

        if i % eval_every == 0
            mean_return = mean(evaluate(mdp, Q, n_episodes=100  00))
            push!(curve_xs, cumulative_steps)
            push!(curve_ys, mean_return)

            if mean_return > best_return
                best_return = mean_return
                best_Q = copy(Q)
                println("  ↑ New best Q-table saved (return: $(round(best_return, digits=4)))")
            end

        end

        if i % (n_episodes ÷ 10) == 0
            println("Episode $i / $n_episodes, ϵ=$(round(ϵ, digits=3))")
        end   
    end

    println("\nBest Q-table achieved mean return: $(round(best_return, digits=4))")
    return best_Q, curve_xs, curve_ys
end

# ----------- EVALUATION (SHOE-AWARE) ----------- #
function evaluate(mdp, Q; n_episodes=10000)
    function greedy_policy(m, s)
        key = q_key(s)
        q_vals = [get(Q, (key, a), 0.0) for a in ACTIONS]
        return ACTIONS[argmax(q_vals)]
    end

    shoe = Shoe(NUM_DECKS)
    returns = Float64[]

    for _ in 1:n_episodes
        push!(returns, rollout(mdp, greedy_policy, shoe))
    end

    return returns
end

# ----------- LEARNING CURVE ----------- #
function plot_learning_curve(curve_xs, curve_ys)
    p = plot(
        curve_xs, curve_ys,
        xlabel="Steps in environment",
        ylabel="Avg return",
        label="Q-Learning",
        legend=:bottomright
    )
    return p
end

# ----------- RUN Q-LEARNING ----------- #
num_episodes = 100000
println("\nRunning Q-Learning for $num_episodes episodes...")
final_Q, curve_xs, curve_ys = qlearning!(m_onedeck, n_episodes=num_episodes, α=0.2, eval_every=100)

p_steps = plot_learning_curve(curve_xs, curve_ys)
display(p_steps)

# ----------- FINAL COMPARISON ----------- #
println("Evaluating final Q-learning policy...")
q_scores = evaluate(m_onedeck, final_Q, n_episodes=10000)

mean_baseline  = mean(baseline_score)
se_baseline    = std(baseline_score)    / sqrt(length(baseline_score))
mean_always    = mean(baseline_always_hit_score)
se_always      = std(baseline_always_hit_score) / sqrt(length(baseline_always_hit_score))
mean_q         = mean(q_scores)
se_q           = std(q_scores)          / sqrt(length(q_scores))

println("\n" * "="^40)
println("          POLICY COMPARISON RESULTS      ")
println("="^40)
println("Baseline (Always hit):")
println("  Mean Return: ", round(mean_always,   digits=4))
println("  Std Error:   ", round(se_always,     digits=4))
println("-"^40)
println("Baseline (Stick on 17+):")
println("  Mean Return: ", round(mean_baseline, digits=4))
println("  Std Error:   ", round(se_baseline,   digits=4))
println("-"^40)
println("Q-Learning:")
println("  Mean Return: ", round(mean_q,        digits=4))
println("  Std Error:   ", round(se_q,          digits=4))
println("="^40)