import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_confusion_matrix(filename, output_filename):
    data = [json.loads(line) for line in open(filename)]
    print(len(data))

    all_true_labels = []
    all_pred_labels = []

    classes = json.load(open('../data/cars_classes.json'))

    accs1 = []
    accs5 = []
    for item in data:
        probs = []
        for idx, prob in enumerate(item["probs"]):
            probs.append((np.array(prob)).mean())

        preds = np.array(probs).argsort()[-5:][::-1]
        label = int(item["label_index"])


        all_true_labels.append(label)
        all_pred_labels.append(preds[0])

        accs1.append(preds[0] == label)
        accs5.append(label in preds)

    print("mean")
    print(np.array(accs1).mean() * 100, np.array(accs5).mean() * 100)

    accs1 = []
    accs5 = []
    for item in data:
        probs = []
        for idx, prob in enumerate(item["probs"]):
            probs.append((np.array(prob)).sum())

        preds = np.array(probs).argsort()[-5:][::-1]
        label = int(item["label_index"])

        all_true_labels.append(label)
        all_pred_labels.append(preds[0])

        accs1.append(preds[0] == label)
        accs5.append(label in preds)

    print("sum")
    print(np.array(accs1).mean() * 100, np.array(accs5).mean() * 100)

    all_probs = []
    for item in data:
        all_probs.append(item["probs"])
    print(len(all_probs[0]))

    calibrated_probs = []
    for i in range(len(all_probs[0])):
        probs = []
        for item in all_probs:
            probs.append(item[i])
        calibrated_probs.append(np.mean(probs, axis=0))
    print(len(calibrated_probs))
    scale_probs = []
    for i in range(len(all_probs[0])):
        probs = []
        for item in all_probs:
            probs.append(item[i])
        scale_probs.append(np.max(np.sum(probs,axis=1), axis=0) - np.min(np.sum(probs,axis=1),axis=0))
    print(len(scale_probs))


    for alpha in range(8, 9):
        accs1 = []
        accs5 = []
        for item in data:
            probs = []
            for idx, prob in enumerate(item["probs"]):
                #if idx != 150 and idx != 151:
                if False:
                    probs.append(-100000)
                else:
                    probs.append(
                        (((
                            np.array(prob)
                            - np.array(calibrated_probs[idx]))/scale_probs[idx])
                        ).sum()
                    )

            preds = np.array(probs).argsort()[-5:][::-1]
            label = int(item["label_index"])
            #if label == 130 or label == 131:
            if True:
                all_true_labels.append(label)
                all_pred_labels.append(preds[0])

                accs1.append(preds[0] == label)
                accs5.append(label in preds)

        print(alpha, np.array(accs1).mean() * 100, np.array(accs5).mean() * 100)

    # Generate confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels = range(196))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(output_filename)  # Save the confusion matrix plot
    plt.close()

    # Print top confusion labels
    cm_flat = cm.flatten()
    top_confusions = np.argsort(-cm_flat)[:200]  # Get top 5 confusions

    print("Top confusion labels:")
    for idx in top_confusions:
        true_label_idx = int(idx // cm.shape[1])
        pred_label_idx = int(idx % cm.shape[1])
        true_label = classes[int(idx // cm.shape[1])]
        pred_label = classes[int(idx % cm.shape[1])]
        #if True:
        if true_label_idx != pred_label_idx:
            print(f"True label: {true_label_idx}, Predicted label: {pred_label_idx}, Count: {cm[true_label_idx, pred_label_idx]}")
            print(f"True label: {true_label}, Predicted label: {pred_label}, Count: {cm[true_label_idx, pred_label_idx]}")

# Example usage
get_confusion_matrix('outputs/cars_idefics_10_cot_seqi4.jsonl', 'output_confusion_matrix.png')

