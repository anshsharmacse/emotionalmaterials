"""
PolyMind AI — Neural Network Model (TensorFlow/Keras)
Multi-Layer Perceptron for Mental State Prediction
Author: Ansh Sharma | B230825MT

Architecture:
  Input(6)  →  Dense(128, ReLU) + BN + Dropout(0.2)
            →  Dense(64,  ReLU) + Dropout(0.15)
            →  Dense(32,  ReLU)
            →  Softmax(4)  [Stressed | Calm | Anxious | Focused]
"""

import numpy as np
import os, json

# ── Mental state labels ────────────────────────────────────────────────────
STATES = ["Stressed", "Calm", "Anxious", "Focused"]
STATE_COLORS = ["#f72585", "#00d4ff", "#f9c74f", "#39d353"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "mental_state_model.h5")

_model = None  # cached model

def build_model():
    """Build and return the Keras MLP model."""
    try:
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Dense(128, activation="relu", input_shape=(6,),
                               name="input_layer"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(64, activation="relu", name="hidden_1"),
            keras.layers.Dropout(0.15),

            keras.layers.Dense(32, activation="relu", name="hidden_2"),

            keras.layers.Dense(4, activation="softmax", name="output"),
        ], name="mental_state_mlp")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    except ImportError:
        return None


def generate_synthetic_dataset(n: int = 10000):
    """
    Generate a physically-consistent training dataset.
    Maps (strain, conductivity, temperature, band_gap, heart_rate, gsr) → state.

    State rules (simplified physics):
      Stressed  : high strain + high HR + high GSR
      Calm      : low strain + normal HR + low GSR
      Anxious   : moderate strain + elevated HR + very high GSR
      Focused   : low strain + normal HR + low GSR + high conductivity
    """
    rng = np.random.default_rng(42)
    X, y = [], []
    per_class = n // 4

    def clip_noise(val, lo, hi, noise=0.05):
        return float(np.clip(val + rng.normal(0, noise*(hi-lo)), lo, hi))

    for _ in range(per_class):  # Stressed
        X.append([clip_noise(0.12, 0.05, 0.20),   # strain
                  clip_noise(400, 200, 600),        # conductivity
                  clip_noise(308, 300, 316),        # temperature
                  clip_noise(2.1, 1.8, 2.5),        # band_gap
                  clip_noise(110, 90, 140),          # heart_rate
                  clip_noise(75, 55, 100)])          # gsr
        y.append(0)

    for _ in range(per_class):  # Calm
        X.append([clip_noise(0.03, 0.0, 0.06),
                  clip_noise(850, 700, 1000),
                  clip_noise(300, 296, 304),
                  clip_noise(1.42, 1.3, 1.55),
                  clip_noise(65, 55, 75),
                  clip_noise(15, 5, 30)])
        y.append(1)

    for _ in range(per_class):  # Anxious
        X.append([clip_noise(0.08, 0.04, 0.14),
                  clip_noise(550, 350, 750),
                  clip_noise(304, 299, 309),
                  clip_noise(1.75, 1.5, 2.1),
                  clip_noise(95, 75, 120),
                  clip_noise(85, 65, 100)])
        y.append(2)

    for _ in range(per_class):  # Focused
        X.append([clip_noise(0.04, 0.01, 0.07),
                  clip_noise(780, 650, 920),
                  clip_noise(301, 298, 306),
                  clip_noise(1.55, 1.35, 1.75),
                  clip_noise(72, 60, 82),
                  clip_noise(25, 10, 40)])
        y.append(3)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_model(epochs: int = 50, save_path: str = MODEL_PATH) -> dict:
    """Train the MLP on synthetic data and save weights."""
    model = build_model()
    if model is None:
        return {"error": "TensorFlow not installed"}

    import tensorflow as tf
    X, y = generate_synthetic_dataset(10000)
    y_cat = tf.keras.utils.to_categorical(y, 4)

    # Feature normalisation (save stats for inference)
    mu = X.mean(axis=0); std = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / std

    split = int(0.85 * len(X))
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y_cat[:split], y_cat[split:]

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=64, callbacks=cb, verbose=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    # Save normalisation params
    norm_path = save_path.replace(".h5", "_norm.json")
    with open(norm_path, "w") as f:
        json.dump({"mu": mu.tolist(), "std": std.tolist()}, f)

    val_acc = max(history.history.get("val_accuracy", [0]))
    return {
        "val_accuracy": round(float(val_acc), 4),
        "epochs_run": len(history.history["loss"]),
        "train_history": {
            "loss":     [round(v,4) for v in history.history["loss"]],
            "val_loss": [round(v,4) for v in history.history.get("val_loss",[])],
            "acc":      [round(v,4) for v in history.history.get("accuracy",[])],
            "val_acc":  [round(v,4) for v in history.history.get("val_accuracy",[])],
        }
    }


def load_model():
    """Load or initialize the model."""
    global _model
    if _model is not None:
        return _model
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
        else:
            _model = build_model()
            if _model:
                train_model()
                _model = tf.keras.models.load_model(MODEL_PATH)
    except ImportError:
        _model = None
    return _model


def _load_norm_params():
    norm_path = MODEL_PATH.replace(".h5", "_norm.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            d = json.load(f)
        return np.array(d["mu"]), np.array(d["std"])
    # Default normalisation stats from the synthetic dataset
    return (np.array([0.07, 625.0, 302.5, 1.73, 86.0, 50.0]),
            np.array([0.05, 225.0,   4.5,  0.32, 22.0, 33.0]))


def predict_mental_state(features: np.ndarray) -> dict:
    """
    Predict mental state from 6 polymer/biometric features.
    features shape: (1, 6) — [strain, conductivity, temperature,
                               band_gap, heart_rate, gsr_signal]
    """
    mu, std = _load_norm_params()
    X_norm = (features - mu) / std

    model = load_model()
    if model is not None:
        probs = model.predict(X_norm, verbose=0)[0].tolist()
    else:
        # Physics-based heuristic when TF not available
        s, c, t, bg, hr, gsr = features[0]
        stress = (s/0.20)*0.35 + (hr/150)*0.30 + (gsr/100)*0.35
        calm   = max(0, (1 - stress) * (c/1000) * 0.9)
        anx    = (gsr/100)*0.5 + (s/0.20)*0.25 + max(0, hr-80)/70*0.25
        focus  = max(0, (c/1000)*0.4 + (1-gsr/100)*0.3 + (1-s/0.20)*0.3)
        raw    = np.array([stress, calm, anx, focus])
        probs  = (raw / raw.sum()).tolist()

    max_idx = int(np.argmax(probs))
    return {
        "state":         STATES[max_idx],
        "probabilities": {STATES[i]: round(p, 4) for i, p in enumerate(probs)},
        "confidence":    round(float(probs[max_idx]), 4),
        "color":         STATE_COLORS[max_idx],
    }


def get_architecture_summary() -> dict:
    """Return model architecture for frontend visualization."""
    return {
        "layers": [
            {"name": "Input",     "units": 6,   "activation": "—",
             "features": ["Strain","Conductivity","Temperature","Band Gap","Heart Rate","GSR"]},
            {"name": "Dense 128", "units": 128, "activation": "ReLU",
             "extras": ["BatchNorm", "Dropout(0.2)"]},
            {"name": "Dense 64",  "units": 64,  "activation": "ReLU",
             "extras": ["Dropout(0.15)"]},
            {"name": "Dense 32",  "units": 32,  "activation": "ReLU"},
            {"name": "Output",    "units": 4,   "activation": "Softmax",
             "classes": STATES},
        ],
        "total_params": 128*6 + 128 + 64*128 + 64 + 32*64 + 32 + 4*32 + 4,
        "optimizer": "Adam(lr=1e-3)",
        "loss": "Categorical Crossentropy",
    }
