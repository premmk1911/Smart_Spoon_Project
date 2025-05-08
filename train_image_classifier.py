# File: scripts/train_image_classifier.py

import os
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def main(
    data_dir="data/images",
    model_path="models/dish_classifier.h5",
    ci_path="models/class_indices.pkl",
    img_size=(224, 224),
    batch_size=16,
    epochs=30,
    lr=1e-4
):
    # 1) Ensure model output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 2) Set up data generators (80/20 split, heavy augmentation)
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        subset="training",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        subset="validation",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # 3) Build the model
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    outputs = Dense(train_gen.num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])

    # 4) Prepare callbacks
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    earlystop  = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

    # 5) Train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, earlystop, reduce_lr]
    )

    # 6) Save class_indices right after training finishes
    class_indices = train_gen.class_indices
    joblib.dump(class_indices, ci_path)
    print(f"✅ Saved class_indices to {ci_path}")

    # 7) Final confirmation
    print(f"✅ Model training complete.")
    print(f"  • Weights: {model_path}")
    print(f"  • Class mapping: {ci_path}\n  {class_indices}")

if __name__ == "__main__":
    main()
