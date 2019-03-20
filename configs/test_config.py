class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False
    high             = 255.

    # output config
    output_path  = "results/DQN_debugging/"
    model_output = output_path + "model.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 20
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 500
    log_freq          = 1
    eval_freq         = 1000
    soft_epsilon      = 0

    #hyper params
    nsteps_train       = 1000000
    batch_size         = 64
    # buffer_size        = 1000
    target_update_freq = 100    
    gamma              = 0.9
    # learning_freq      = 4
    # state_history      = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = nsteps_train/2
    # learning_start     = 200

    state_shape = (16, 8, 4)
    num_layers         = 3
    hidden_size        = 1024
