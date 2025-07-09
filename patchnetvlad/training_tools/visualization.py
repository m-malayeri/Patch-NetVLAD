# patchnetvlad/training_tools/visualization.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

def visualize_recalls(predictions, gt, whole_test_set, dataset_root="/home/jovyan/data/VPR/LW", position_file="/home/jovyan/data/VPR/LW/image_pose_place.txt"):
    positions = {}
    with open(position_file, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("image_path"):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                image_path = parts[0]
                # image_path in positions Database/task_96c959ea-db3c-4c49-a09f-84d4c35b4911/1747991634_r.jpg
                x = float(parts[1])
                y = float(parts[2])
                positions[image_path] = (x, y)

    print("====> Visualizing recalls for the first 10 query images")
    num_to_visualize = 10
    top_k = 5
    localization_errors = []

    for qIx in range(num_to_visualize):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()
        query_path = whole_test_set.qImages[qIx]
        query_path = '/'.join(query_path.rsplit('/', 3)[-3:])
        fig.suptitle(f"Query #{qIx + 1}: {query_path}", fontsize=13)

        q_path = os.path.join(dataset_root, query_path)
        if os.path.exists(q_path):
            q_img = mpimg.imread(q_path)
            axes[0].imshow(q_img)
            axes[0].set_title("Query", fontsize=11)
        else:
            axes[0].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=10)
        axes[0].axis('off')

        # query_path Query/task_6cbb51cf-7033-40bf-af18-6d0489f983e8/1747740419_f.jpg
        # q_path /home/jovyan/data/VPR/LW/Query/task_6cbb51cf-7033-40bf-af18-6d0489f983e8/1747740419_f.jpg
        q_pos = positions.get(query_path, (None, None))
        pred = predictions[qIx][:top_k]
        true_positives = set(gt[qIx])

        if q_pos != (None, None) and len(pred) > 0:
            top1_idx = pred[0]
            db_rel_path = whole_test_set.dbImages[top1_idx]
            db_rel_path = '/'.join(db_rel_path.rsplit('/', 3)[-3:])
            db_pos = positions.get(db_rel_path, (None, None))
            if db_pos != (None, None):
                dx = db_pos[0] - q_pos[0]
                dy = db_pos[1] - q_pos[1]
                error = math.sqrt(dx**2 + dy**2)
                localization_errors.append(error)

        for i, db_idx in enumerate(pred):
            ax = axes[i + 1]
            db_rel_path = whole_test_set.dbImages[db_idx]
            db_rel_path = '/'.join(db_rel_path.rsplit('/', 3)[-3:])
            db_path = os.path.join(dataset_root, db_rel_path)

            db_pos = positions.get(db_rel_path, (None, None))
            if None not in (q_pos + db_pos):
                dx, dy = db_pos[0] - q_pos[0], db_pos[1] - q_pos[1]
                dist = math.sqrt(dx**2 + dy**2)
                dist_str = f"{dist:.2f}m"
            else:
                dist_str = "N/A"

            if os.path.exists(db_path):
                db_img = mpimg.imread(db_path)
                ax.imshow(db_img)
                match = "✓" if db_idx in true_positives else "✗"
                color = 'green' if db_idx in true_positives else 'red'
                ax.set_title(f"Top {i+1}: {match}", color=color, fontsize=9)

                path_parts = db_rel_path.split("/")
                if len(path_parts) >= 2:
                    short_path = f"{'/'.join(path_parts[:-1])}/\n{path_parts[-1]}"
                else:
                    short_path = db_rel_path

                label_text = f"{short_path}\n{dist_str}"
                ax.text(0.5, -0.05, label_text, ha='center', va='top',
                        transform=ax.transAxes, fontsize=7, wrap=True)
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=9)

            ax.axis('off')

        for j in range(top_k + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    if localization_errors:
        mean_error = sum(localization_errors) / len(localization_errors)
        print(f"\n====> Mean Top-1 Localization Error: {mean_error:.2f} meters")
    else:
        print("\n====> No valid position data to compute localization error.")
