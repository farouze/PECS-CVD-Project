import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..test import cfd_test_predictions, cfd_test_full_predictions
from ..report import CFDDiscoveryReport
from ..utils import validate_domains

def _require_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
        return tf, layers, models, callbacks
    except ImportError as e:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow") from e

def _prepare_domain_fit(X):
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs[..., np.newaxis], scaler

def _prepare_domain_transform(X, scaler):
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    Xs = scaler.transform(X)
    return Xs[..., np.newaxis]

def default_1d_cnn_encoder_classifier(input_shape, num_classes=2, latent_dim=128, learning_rate=1e-3):
    tf, layers, models, callbacks = _require_tf()
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    z = layers.Dense(latent_dim, activation="relu", name="latent_embedding")(x)
    x = layers.Dropout(0.3)(z)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def _embedding_model(model):
    tf, layers, models, callbacks = _require_tf()
    return models.Model(inputs=model.input, outputs=model.get_layer("latent_embedding").output)

def _fusion_classifier(input_dim, num_classes=2, learning_rate=1e-3):
    tf, layers, models, callbacks = _require_tf()
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def _train(model, X, y, validation_split, epochs, batch_size, patience, verbose):
    tf, layers, models, callbacks = _require_tf()
    early = callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)
    model.fit(
        X, y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=verbose,
    )
    return model

def _predict(model, X):
    probs = model.predict(X, verbose=0)
    return probs, np.argmax(probs, axis=1)

def cfd_same_encoder_discover(
    domains,
    y,
    encoder_builder=None,
    task="classification",
    metric="accuracy",
    test_size=0.25,
    random_state=42,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    patience=8,
    verbose=1,
    alpha=0.05,
    min_gain=0.01,
    n_bootstrap=200,
    n_permutations=200,
    run_full=True,
):
    if task != "classification":
        raise NotImplementedError("Same-encoder deep CFD currently supports classification only")

    domains, y = validate_domains(domains, y)
    names = list(domains.keys())
    y = np.asarray(y)
    num_classes = len(np.unique(y))

    if encoder_builder is None:
        encoder_builder = default_1d_cnn_encoder_classifier

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y)

    y_train = y[train_idx]
    y_test = y[test_idx]

    domain_preds = {}
    embeddings_train = {}
    embeddings_test = {}

    for name in names:
        print(f"\n[pyCFD] Training domain encoder: {name}")
        X_train, scaler = _prepare_domain_fit(domains[name][train_idx])
        X_test = _prepare_domain_transform(domains[name][test_idx], scaler)

        model = encoder_builder(input_shape=X_train.shape[1:], num_classes=num_classes)
        model = _train(model, X_train, y_train, validation_split, epochs, batch_size, patience, verbose)

        _, pred = _predict(model, X_test)
        domain_preds[name] = pred

        emb_model = _embedding_model(model)
        embeddings_train[name] = emb_model.predict(X_train, verbose=0)
        embeddings_test[name] = emb_model.predict(X_test, verbose=0)

        print(f"[pyCFD] {name} accuracy: {np.mean(pred == y_test):.4f}")

    pairwise_results = {}
    rows = []
    matrix = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)

    for a, b in itertools.combinations(names, 2):
        print(f"\n[pyCFD] Training fusion: {a} + {b}")
        Z_train = np.concatenate([embeddings_train[a], embeddings_train[b]], axis=1)
        Z_test = np.concatenate([embeddings_test[a], embeddings_test[b]], axis=1)

        fusion_model = _fusion_classifier(Z_train.shape[1], num_classes=num_classes)
        fusion_model = _train(fusion_model, Z_train, y_train, validation_split, epochs, batch_size, patience, verbose)
        _, fusion_pred = _predict(fusion_model, Z_test)

        res = cfd_test_predictions(
            y_true=y_test,
            y_a=domain_preds[a],
            y_b=domain_preds[b],
            y_fusion=fusion_pred,
            name_a=a,
            name_b=b,
            task=task,
            metric=metric,
            alpha=alpha,
            min_gain=min_gain,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        res.type = "same_encoder_pairwise"
        res.metadata = {"encoder_rule": "same architecture, separate weights", "fusion": "latent_embedding_concat"}

        pairwise_results[f"{a}_vs_{b}"] = res
        matrix.loc[a, b] = res.cfd_index
        matrix.loc[b, a] = res.cfd_index

        rows.append({
            "combination": f"{a}_vs_{b}",
            "domains": f"{a}+{b}",
            "type": res.type,
            "cfd_index": res.cfd_index,
            "fusion_gain": res.fusion_gain,
            "p_value": res.p_value,
            "ci_lower": res.confidence_interval[0],
            "ci_upper": res.confidence_interval[1],
            "redundancy": res.redundancy,
            "complementary": res.is_complementary,
        })

    full_result = None
    if run_full and len(names) >= 3:
        print(f"\n[pyCFD] Training full fusion: {' + '.join(names)}")
        Z_train = np.concatenate([embeddings_train[n] for n in names], axis=1)
        Z_test = np.concatenate([embeddings_test[n] for n in names], axis=1)

        fusion_model = _fusion_classifier(Z_train.shape[1], num_classes=num_classes)
        fusion_model = _train(fusion_model, Z_train, y_train, validation_split, epochs, batch_size, patience, verbose)
        _, full_pred = _predict(fusion_model, Z_test)

        full_result = cfd_test_full_predictions(
            y_true=y_test,
            y_domain_preds=domain_preds,
            y_fusion=full_pred,
            task=task,
            metric=metric,
            alpha=alpha,
            min_gain=min_gain,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        full_result.type = "same_encoder_full_fusion"
        full_result.metadata = {
            "encoder_rule": "same architecture, separate weights",
            "fusion": "latent_embedding_concat",
            "full_redundancy": "mean_pairwise_prediction_redundancy",
        }

        rows.append({
            "combination": "FULL_FUSION",
            "domains": "+".join(names),
            "type": full_result.type,
            "cfd_index": full_result.cfd_index,
            "fusion_gain": full_result.fusion_gain,
            "p_value": full_result.p_value,
            "ci_lower": full_result.confidence_interval[0],
            "ci_upper": full_result.confidence_interval[1],
            "redundancy": full_result.redundancy,
            "complementary": full_result.is_complementary,
        })

    ranking = pd.DataFrame(rows).sort_values(
        ["complementary", "cfd_index", "fusion_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    if ranking.empty or not bool(ranking.iloc[0]["complementary"]):
        recommendation = "No statistically supported complementary combination detected."
    else:
        top = ranking.iloc[0]
        recommendation = (
            f"Recommended fusion: {top['combination']} ({top['domains']}). "
            f"CFD Index={top['cfd_index']:.2f}, Fusion Gain={top['fusion_gain']:.4f}, p={top['p_value']:.6f}."
        )

    return CFDDiscoveryReport(
        pairwise_results=pairwise_results,
        full_result=full_result,
        matrix=matrix,
        ranking=ranking,
        recommended_strategy=recommendation,
        metadata={
            "mode": "same_encoder_deep",
            "encoder_rule": "same architecture, separate weights",
            "task": task,
            "metric": metric,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
    )
