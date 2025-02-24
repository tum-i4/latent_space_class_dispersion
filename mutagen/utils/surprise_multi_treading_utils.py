def cal_dsa_worker(train_ats, train_pred, layer_output, target_pred, target_ats):

    if layer_output[0].ndim == 3:  # For convolutional layers
        layer_matrix = np.array(torch.mean(layer_output, dim=(-1, -2)))
    else:  # For fcn layers
        layer_matrix = np.array(layer_output)

    if target_ats is None:
        target_ats = layer_matrix
    else:
        target_ats = torch.cat((target_ats, layer_matrix), dim=1)
        # target_ats = np.append(target_ats, layer_matrix, axis=1)
        layer_matrix = None

    # Fetch Distance-based surprise adequacy
    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    dsa_list_img_wise = []

    # print("Fetching DSA")
    # Convert target_pred to numpy array
    for i, at in enumerate(target_ats):
        label = target_pred  # [idx]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[int(label)]])
        b_dist, _ = find_closest_at(
            a_dot, train_ats[list(set(all_idx) - set(class_matrix[int(label)]))]
        )
        dsa_list_img_wise.append(a_dist / b_dist)

    return dsa_list_img_wise

def cal_dsa(train_act_traces, layers_outputs, output_classes):
    """Surprise-adequacy coverage for 1 input
    Args:
        layers_outputs(OrderedDict): The outputs of internal layers for a batch of mutants
    Returns:
        ptr(Tensor): array that records the coverage information
    """
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ats = train_act_traces["AT"]  # (#samples, 10 (op_layers))
    train_pred = train_act_traces["predictions"]  # (#samples,)

    # dsa_list_img_wise = []
    with mp.Pool(processes=8) as pool:
        results = []
        for idx in tqdm(range(len(output_classes))):
            layer_output = torch.tensor([layers_outputs[idx]])
            # layers_output = layers_outputs[idx]
            target_pred = output_classes[idx]

            # CALCULATE TARGET PREDICTION
            target_ats = None
            target_pred = output_classes[idx]

            result = pool.apply_async(cal_dsa_worker, (train_ats, train_pred, layer_output, target_pred, target_ats))
            results.append(result)
        
        dsa_list_img_wise = []
        for result in tqdm(results, desc="Processing results"):
            dsa_list_img_wise.extend(result.get())

    # dsa_img = np.array([dsa_list_img_wise])

    return dsa_list_img_wise