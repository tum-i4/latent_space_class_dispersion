import random

import numpy as np
from tqdm import tqdm

from fuzzer.util import predict, save_seeds, save_seeds_gray, save_seeds_rgb


def compare_and_append_indices(gt_labels, pred_labels, selected_class, condition):
    selected_idx = [
        index
        for index, (a, b) in enumerate(zip(gt_labels, pred_labels))
        if all(eval(condition, globals(), {"a": a, "b": b, "c": selected_class}) for condition in condition)
    ]

    return selected_idx


def compare_and_append_indices_prob(gt_labels, pred_labels, selected_class, op_prob, condition):
    selected_idx = [
        index
        for index, (a, b, d) in enumerate(zip(gt_labels, pred_labels, op_prob))
        if all(eval(condition, globals(), {"a": a, "b": b, "c": selected_class, "d": d}) for condition in condition)
    ]

    return selected_idx


class ClassificationNetworks:
    def __init__(self, seed_selection_strategy, num_classes=None):
        self.num_classes = num_classes
        self.seed_selection_strategy = seed_selection_strategy

    def seed_selection(self, model, test_loader, output_path, config, partial_dataset=False, num_samples_per_class=50):
        """
        seed selection for classification networks
        :param model: model to be tested
        :param test_loader: of the dataset to be tested on
        :param output_path: dir to save seeds in
        :param config: configuration settings
        :return:
        """
        if not config.fuzz_entire_test_set:
            gt_labels, pred_labels, pred_op_prob, pred_probs = model.good_seed_detection(test_loader, predict, config)

            selected_idx = []

            img_ids = np.array(test_loader.image_ids)
            img_paths = np.array(test_loader._image_paths)

            if partial_dataset:
                if config.init_seed_selection == "true_classified":
                    for selected_class in range(config.num_classes):
                        condition = ["a == b == c"]  # condition = [lambda a, b, c: a == b == c]
                        selected_idx_cl = compare_and_append_indices(gt_labels, pred_labels, selected_class, condition)
                        random.shuffle(selected_idx_cl)
                        selected_idx.extend(selected_idx_cl[:num_samples_per_class])
                        # print(len(selected_idx_cl), len(selected_idx))

                elif config.init_seed_selection == "high_prob":
                    condition = ["a == b == c and (d > 0.5)"]
                    selected_idx_cl = compare_and_append_indices_prob(
                        gt_labels, pred_labels, selected_class, pred_op_prob, condition
                    )
                    random.shuffle(selected_idx_cl)
                    selected_idx.extend(selected_idx_cl[:num_samples_per_class])

                else:
                    raise NotImplementedError("Please extend initial seed selection strategy for this network type.")

            else:
                if config.init_seed_selection == "true_classified":
                    selected_idx = np.array(gt_labels) == np.array(pred_labels)

                elif config.init_seed_selection == "high_prob":
                    selected_idx = (np.array(gt_labels) == np.array(pred_labels)) & (np.array(pred_op_prob) > 0.5)

                else:
                    raise NotImplementedError("Please extend initial seed selection strategy for this network type.")

            selected_img_ids = img_ids[selected_idx]
            selected_img_paths = img_paths[selected_idx]

            print("Total test images:", img_ids.shape[0])
            print("Correctly Classified test images:", (np.array(gt_labels) == np.array(pred_labels)).sum())
            print("Stored test images for initial seeds:", selected_img_ids.shape[0])

            if self.seed_selection_strategy == "true_classified" or self.seed_selection_strategy == "high_prob":
                if config.data == "gtsrb-gray" or config.data == "mnist":
                    for i in tqdm(range(selected_img_ids.shape[0])):
                        save_seeds_gray(
                            selected_img_ids[i], selected_img_paths[i], output_path
                        )  # img_path is not string as its image here...

                else:
                    for i in tqdm(range(selected_img_ids.shape[0])):
                        save_seeds_rgb(selected_img_ids[i], str(selected_img_paths[i]), output_path)
            else:
                raise NotImplementedError(
                    "This Seed selection strategy for Classification Networks to be implemented."
                )

        else:
            print("Fuzzer uses entire test dataset so initial seeds will be used from data source directory.")

    def objective_function(self, seed):
        """
        checks if a CorpusElement satisifies the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
        o check whether it is an adversarial sample. crashes records the suffix for the name of the failed tests
        :param seed:
        :return:
        """
        crash_rec, crash_f1, crash_hyb = [], [], []
        if seed.predictions["op_class"] != seed.ground_truths["op_class"]:
            crash_rec.append("")
        return crash_rec, crash_f1, crash_hyb
