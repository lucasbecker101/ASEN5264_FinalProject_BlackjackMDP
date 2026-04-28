using POMDPs
using POMDPTools: SparseCat, Deterministic
using QuickPOMDPs: QuickPOMDP
using Flux
using Statistics
using Random

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
# DEALER (FIXED — NO SHADOWING sum)
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

# correct dealer initialization
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
             dealer_sum,   # store final dealer result
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
    # 1. If we are ALREADY in a terminal state, no more rewards are given.
    if s.terminal
        return 0.0
    end

    # 2. If the NEXT state is not terminal, no reward is given yet.
    if !sp.terminal
        return 0.0
    end

    # 3. Calculate final win/loss/draw
    if sp.player_sum > 21
        return -1.0
    end

    dealer_sum = sp.dealer_upcard  # overloaded as final result

    if dealer_sum > 21 || sp.player_sum > dealer_sum
        return 1.0
    elseif sp.player_sum == dealer_sum
        return 0.0
    else
        return -1.0
    end
end

############################
# INITIAL STATE
############################
function build_initial_dist()
    init_deck = (4,4,4,4,4,4,4,4,4,16)
    total_cards = sum(init_deck)
    
    states = BJState[]
    probs = Float64[]
    
    for card in 1:10
        # Probability of drawing this card as the upcard
        p = init_deck[card] / total_cards 
        
        # Remove it from the deck
        new_deck = remove_card(init_deck, card)
        
        # Player sum starts at 0, dealer has their upcard
        push!(states, BJState(0, card, false, new_deck, false))
        push!(probs, p)
    end
    
    return SparseCat(states, probs)
end

initial_dist = build_initial_dist()

############################
# MDP
############################
############################
# MDP
############################
m_onedeck = QuickPOMDP(
    actions = ACTIONS,
    discount = 0.995,

    transition = bj_transition,
    reward = bj_reward,
    initialstate = initial_dist,

    observation = (a, sp) -> Deterministic(sp),
    obstype = BJState,
    
    # ADD THIS LINE:
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
# ROLLOUT
############################
function rollout(mdp, policy_function, s0, max_steps)
    r_total = 0.0
    disc = 1.0
    γ = discount(mdp)
    s = s0

    for t in 1:max_steps
        if isterminal(mdp, s)
            break
        end

        a = policy_function(mdp, s)

        dist = bj_transition(s, a)
        sp = rand(dist)

        r = bj_reward(s, a, sp)

        r_total += disc * r
        disc *= γ
        s = sp
    end

    return r_total
end

############################
# SIMULATION
############################
baseline_score = [
    rollout(m_onedeck, baseline_policy, rand(initial_dist), 10)
    for _ in 1:10000
]

println(baseline_score[1:25])  # print first 10 rewards for sanity check
println("Mean reward: ", mean(baseline_score))
println("Std error: ", std(baseline_score) / sqrt(length(baseline_score)))

# Map actions to indices for network outputs
const ACTION_TO_IDX = Dict(:hit => 1, :stick => 2)
const IDX_TO_ACTION = Dict(1 => :hit, 2 => :stick)

function state_to_vec(s::BJState)
    # Calculate total remaining cards to find probabilities
    total_remaining = max(1, sum(s.deck_counts)) 
    norm_counts = [c / total_remaining for c in s.deck_counts]
    
    return Float32.([
        s.player_sum / 21.0,       # Normalize to max possible non-bust value
        s.dealer_upcard / 10.0,    # Normalize to max face value
        s.usable_ace ? 1.0 : 0.0, 
        norm_counts...             # Now represents the % chance of drawing each card
    ])
end

mutable struct ReplayBuffer
    capacity::Int
    memory::Vector{NamedTuple}
    pos::Int
end

function ReplayBuffer(capacity::Int)
    return ReplayBuffer(capacity, NamedTuple[], 1)
end

function push_transition!(buffer::ReplayBuffer, transition)
    if length(buffer.memory) < buffer.capacity
        push!(buffer.memory, transition)
    else
        buffer.memory[buffer.pos] = transition
        buffer.pos = mod1(buffer.pos + 1, buffer.capacity)
    end
end

function sample_batch(buffer::ReplayBuffer, batch_size::Int)
    # Randomly sample a batch of transitions
    return rand(buffer.memory, min(batch_size, length(buffer.memory)))
end

function create_q_network(input_size::Int, output_size::Int)
    return Chain(
        Dense(input_size => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => output_size)
    )
end

function train_dqn!(mdp; n_episodes=10000, batch_size=256, capacity=500_000, γ=1.0)
    # 1. Initialize Networks
    input_size = 13 # Length of state_to_vec output
    output_size = 2 # Hit or Stick
    
    q_net = create_q_network(input_size, output_size)
    target_net = deepcopy(q_net) # Copy for stable targets
    
    opt_state = Flux.setup(Adam(5e-5), q_net)
    buffer = ReplayBuffer(capacity)
    
    target_update_freq = 5000
    global_step = 0
    returns = Float64[]

    for ep in 1:n_episodes
        # Setup Epsilon for this episode (Decays from 1.0 to 0.05)
        ϵ = max(0.01, 1.0 - ep / (n_episodes * 0.9))
        
        # Safe Initialization
        init_dist = initialstate(mdp)
        s = init_dist isa Deterministic ? init_dist.val : rand(init_dist)
        ep_reward = 0.0

        while !s.terminal
            global_step += 1
            s_vec = state_to_vec(s)

            # 2. Epsilon-Greedy Action Selection
            if rand() < ϵ
                a = rand(ACTIONS)
            else
                q_vals = q_net(s_vec) # Forward pass
                a = IDX_TO_ACTION[argmax(q_vals)]
            end

            # 3. Environment Step
            sp = rand(transition(mdp, s, a))
            r = reward(mdp, s, a, sp)
            ep_reward += r

            # 4. Store in Replay Buffer
            transition_tuple = (s=s_vec, a=ACTION_TO_IDX[a], r=Float32(r), sp=state_to_vec(sp), term=sp.terminal)
            push_transition!(buffer, transition_tuple)

            # 5. Training Step (if buffer has enough data)
            if length(buffer.memory) >= batch_size
                batch = sample_batch(buffer, batch_size)
                
                # Extract batch arrays
                S_batch = hcat([b.s for b in batch]...)       # 13 x Batch
                Sp_batch = hcat([b.sp for b in batch]...)     # 13 x Batch
                R_batch = Float32.([b.r for b in batch])      # Batch
                Term_batch = Bool.([b.term for b in batch])   # Batch
                A_batch = [b.a for b in batch]                # Batch
                
                # --- DOUBLE DQN UPGRADE ---
                # 1. Use the MAIN network to choose the best action for the next state
                main_next_q = q_net(Sp_batch) 
                best_next_actions = [argmax(main_next_q[:, i]) for i in 1:batch_size]

                # 2. Use the TARGET network to evaluate the value of that chosen action
                target_next_q = target_net(Sp_batch) 
                max_target_q = Float32.([target_next_q[best_next_actions[i], i] for i in 1:batch_size])
                # --------------------------
                
                # Bellman Equation Target
                y = R_batch .+ Float32(γ) .* max_target_q .* .!Term_batch
                
                # Compute gradients and update Q-network
                loss, grads = Flux.withgradient(q_net) do net
                    preds = net(S_batch) 
                    selected_preds = [preds[A_batch[i], i] for i in 1:batch_size]
                    Flux.mse(selected_preds, y)
                end
                
                Flux.update!(opt_state, q_net, grads[1])
            end

            # 6. Update Target Network periodically
            if global_step % target_update_freq == 0
                target_net = deepcopy(q_net)
            end

            s = sp
        end
        push!(returns, ep_reward)
        
        if ep % 500 == 0
            println("Episode $ep | Mean Return (last 500): ", round(mean(returns[end-499:end]), digits=3))
        end
    end
    
    return q_net, returns
end

function evaluate_dqn(mdp, trained_q_net; n_episodes=10_000)
    returns = Float64[]
    
    for _ in 1:n_episodes
        init_dist = initialstate(mdp)
        s = init_dist isa Deterministic ? init_dist.val : rand(init_dist)
        
        ep_r = 0.0
        while !s.terminal
            s_vec = state_to_vec(s)
            q_vals = trained_q_net(s_vec)
            a = IDX_TO_ACTION[argmax(q_vals)]
            
            sp = rand(transition(mdp, s, a))
            ep_r += reward(mdp, s, a, sp)
            s = sp
        end
        push!(returns, ep_r)
    end
    
    return returns
end

# Execution
println("Training DQN...")
trained_model, train_history = train_dqn!(m_onedeck, n_episodes=100_000)

println("Evaluating DQN...")
dqn_scores = evaluate_dqn(m_onedeck, trained_model)
println("DQN Mean Return: ", round(mean(dqn_scores), digits=4))