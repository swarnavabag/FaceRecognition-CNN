import random
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.shared.config import (
    VAL_PROCESSED_DIR,
    BEST_MODEL_PATH
)
from src.reco.model import FaceRecognitionResNet
from src.reco.dataset import val_transform


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    model = FaceRecognitionResNet().to(device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# ============================================================
# VALIDATION IDENTITIES
# ============================================================

def build_validation_identity_map():
    identity_to_images = defaultdict(list)

    val_folders = sorted([
        folder for folder in VAL_PROCESSED_DIR.iterdir()
        if folder.is_dir()
    ])

    for folder in val_folders:
        image_paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if len(image_paths) >= 2:
            identity_to_images[folder.name] = image_paths

    return identity_to_images


# ============================================================
# PAIR GENERATION
# ============================================================

def generate_pairs(identity_to_images, pairs_per_identity=20):
    pairs = []
    identities = list(identity_to_images.keys())

    for identity in identities:
        images = identity_to_images[identity]

        # Positive pairs
        for _ in range(pairs_per_identity):
            img1, img2 = random.sample(images, 2)
            pairs.append((img1, img2, 1))

        # Negative pairs
        for _ in range(pairs_per_identity):
            other_identity = random.choice(identities)

            while other_identity == identity:
                other_identity = random.choice(identities)

            img1 = random.choice(images)
            img2 = random.choice(identity_to_images[other_identity])

            pairs.append((img1, img2, 0))

    return pairs


# ============================================================
# EMBEDDING CACHE
# ============================================================

def build_embedding_cache(model, identity_to_images):
    embedding_cache = {}

    with torch.no_grad():
        for identity, image_paths in tqdm(
            identity_to_images.items(),
            desc="Caching embeddings"
        ):
            for img_path in image_paths:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = val_transform(image)
                image = image.unsqueeze(0).to(device)

                embedding = model(image)
                embedding = embedding.squeeze(0).cpu()

                embedding_cache[str(img_path)] = embedding

    return embedding_cache


# ============================================================
# SIMILARITY
# ============================================================

def compute_similarity(embedding1, embedding2):
    return torch.dot(embedding1, embedding2).item()


def evaluate_pairs(pairs, embedding_cache):
    scores = []
    labels = []

    for path1, path2, label in tqdm(pairs, desc="Evaluating pairs"):
        emb1 = embedding_cache[str(path1)]
        emb2 = embedding_cache[str(path2)]

        score = compute_similarity(emb1, emb2)

        scores.append(score)
        labels.append(label)

    return scores, labels


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(scores, labels):
    best_acc = 0
    best_threshold = None

    for threshold in [x / 100 for x in range(-100, 101)]:
        correct = 0

        for score, label in zip(scores, labels):
            pred = 1 if score > threshold else 0

            if pred == label:
                correct += 1

        acc = correct / len(scores)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


# ============================================================
# ANALYSIS
# ============================================================

def analyze_scores(scores, labels):
    positive_scores = []
    negative_scores = []

    for score, label in zip(scores, labels):
        if label == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    print("\nPositive pairs:", len(positive_scores))
    print("Negative pairs:", len(negative_scores))

    print("\nPositive stats")
    print("Min :", min(positive_scores))
    print("Max :", max(positive_scores))
    print("Mean:", sum(positive_scores) / len(positive_scores))

    print("\nNegative stats")
    print("Min :", min(negative_scores))
    print("Max :", max(negative_scores))
    print("Mean:", sum(negative_scores) / len(negative_scores))


# ============================================================
# MAIN
# ============================================================

def main():
    model = load_model()

    identity_to_images = build_validation_identity_map()
    print("Validation identities:", len(identity_to_images))

    pairs = generate_pairs(
        identity_to_images,
        pairs_per_identity=20
    )

    print("Total pairs:", len(pairs))

    embedding_cache = build_embedding_cache(
        model,
        identity_to_images
    )

    print("Cached embeddings:", len(embedding_cache))

    scores, labels = evaluate_pairs(
        pairs,
        embedding_cache
    )

    analyze_scores(scores, labels)

    threshold, accuracy = find_best_threshold(scores, labels)

    print("\nBest threshold:", threshold)
    print("Best verification accuracy:", accuracy)


if __name__ == "__main__":
    main()