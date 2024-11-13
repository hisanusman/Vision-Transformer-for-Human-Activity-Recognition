# Vision-Transformer-for-Human-Activity-Recognition

Task: Human Action Recognition
Human action recognition in videos involves identifying and categorizing different activities or movements displayed in frames. The Vision Transformer (ViT) model, traditionally used for image classification, will be fine-tuned to recognize actions from short video clips. The model will distinguish among multiple action classes by learning patterns across frames.

The HMDB (Human Motion Database) dataset is a well-known benchmark for human action recognition. It features over 6,800 video clips across 51 action categories, such as running, eating, and waving. This dataset includes real-world, dynamic actions from various sources, providing diverse and challenging scenarios for testing action recognition models.

1. Dataset Preprocessing <br>
 Extracted frames from each video in the HMDB dataset. <br>
 Resized the frames to the input size expected by Vision Transformer (ViT). <br>
 Applied data augmentation techniques to improve generalization, such as cropping, flipping, and normalization.

2. Loading the Vision Transformer Model <br>
 Loaded a pre-trained ViT model suitable for image-based tasks. <br>
 Modified the model’s final layers to match the number of classes in the HMDB dataset. <br>
 Used high-level libraries such as hugging face etc.

3. Setting Up Training Configurations <br>
 Chose an appropriate batch size and number of epochs for effective training. <br>
 Set a suitable learning rate for fine-tuning the model on the dataset.

4. Checkpointing and Early Stopping <br>
 Used checkpointing to save the best-performing model during training. <br>
 Implemented early stopping based on validation performance to avoid overfitting.

5. Model Evaluation <br>
 Evaluated the model’s accuracy on the test set and achieved 95% plus accuracy.
