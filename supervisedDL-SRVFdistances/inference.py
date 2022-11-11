import pickle, numpy as np
from tensorflow.keras.models import load_model

# Load the template DB - this is what we match against
template = pickle.load(open("debug_template.pickle", "rb"))


# Load the intake - this is what is unknown and needs to be recognized
intake = pickle.load(open("test_intake.pickle", "rb"))


# Load our surrogate model for shape distances
model = load_model("OCR")

intake_results = [""] * len(intake)
for i, intake_char in enumerate(intake):
    min_dist = np.inf
    for (
        template_char
    ) in (
        template
    ):  # FIXME: Much more efficient to accumulate each pairs per intake and then feed it to predict as a batch
        intake_arr = np.asarray(intake_char["c__x"])
        template_arr = np.asarray(template_char["c"]["x"])
        # Add the batch dimension
        intake_arr = intake_arr[np.newaxis, ...]
        template_arr = template_arr[np.newaxis, ...]
        dist = model.predict([template_arr, intake_arr])
        print(dist, template_char["char"])
        if dist < min_dist:
            intake_results[i] = template_char["char"]
            min_dist = dist

breakpoint()
