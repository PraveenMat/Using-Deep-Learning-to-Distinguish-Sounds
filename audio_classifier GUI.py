import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import csv
from os.path import basename 
import matplotlib.backends.backend_tkagg as mpl_backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
#global listbox_frame


def load_selected_model(model_name):
    model_path = os.path.expanduser(f"~\Documents\FYP\{model_name}.h5")
    return load_model(model_path)  # Return the loaded model

def model_selection_changed(*args):
    global model, model_description_label  # Add model_description_label to the global variables
    model = load_selected_model(selected_model.get())

    # Update the model description label
    model_description_label.config(text=get_model_description(selected_model.get()))

def update_uploaded_files_label():
    global uploaded_files_label, uploaded_file_paths
    uploaded_files_label.config(text=f"Uploaded files: {len(uploaded_file_paths)}")


def load_label_mapping(csv_file_path):
    label_mapping = {}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].isdigit():
                label_mapping[int(row[0])] = row[2]  # Use the display name instead of the mid value
    return label_mapping


custom_classes_mapping = {
    'Human Sounds': list(range(0, 53)) + list(range(64, 67)),
    'Laughter and Crying': list(range(16, 26)),
    'Animal Sounds': list(range(72, 136)),
    'Music and Musical Instruments': list(range(137, 284)),
    'Nature Sounds': list(range(283, 300)),
    'Vehicles Sounds': list(range(300, 343)),
    'Domestic Sounds and Appliances': list(range(343, 384)),
    'Alarms and Sirens': list(range(384, 402)),
    'Mechanical Sounds and Tools': list(range(402, 427)),
    'Explosions, Gunshots, and Fireworks': list(range(427, 435)),
    'Wood and Glass Sounds': list(range(438, 445)),
    'Liquid Sounds': list(range(445, 458)),
    'Miscellaneous Sounds': list(range(458, 501)),
    'Electronic Sounds and Signals': list(range(501, 526)),
    'Ambience and Environment': [526],
    'Inside Spaces': list(range(506, 509)),
    'Outside Spaces': list(range(509, 511)),
    'Spatial Sound': list(range(511, 515)),
    'Noise Types': list(range(515, 523)),
    'Media Sounds': list(range(523, 527)),
}

def convert_label_mapping_to_custom_classes(label_mapping, custom_classes_mapping):
    new_label_mapping = {}
    for custom_class, class_ranges in custom_classes_mapping.items():
        for class_range in class_ranges:
            if class_range in label_mapping:
                new_label_mapping[class_range] = custom_class
    return new_label_mapping

label_mapping = load_label_mapping(os.path.expanduser("~\Documents\FYP\class_labels_indices.csv"))
label_mapping = convert_label_mapping_to_custom_classes(label_mapping, custom_classes_mapping)

#Model descriptions
def get_model_description(model_name):
    descriptions = {
        "Model 1": "Model 1 is trained on 500 of Image data.",
        "Model 2": "Model 2 is trained on 22,176 of data.",
        "Model 3": "Model 3 is trained on 150,000 of data.",
    }
    return descriptions.get(model_name, "Unknown model")


def predict_label(file_path, selected_model):
    # Load the preprocessed audio data
    preprocessed_data = load_preprocessed_audio_data(file_path)
    
    # Assuming 'model' is the selected Keras model
    model_input_shape = model.layers[0].input_shape[1:]

    num_frames, num_features = preprocessed_data.shape

    # Padding if the input data has fewer frames or features than required
    padding_frames = max(model_input_shape[0], num_frames) - num_frames
    padding_features = max(model_input_shape[1], num_features) - num_features
    preprocessed_data = np.pad(preprocessed_data, ((0, padding_frames), (0, padding_features)))

    # Slicing if the input data has more frames or features than required
    preprocessed_data = preprocessed_data[:model_input_shape[0], :model_input_shape[1]]

    # Adding the required number of channels to the input data
    if len(model_input_shape) == 3:
        required_channels = model_input_shape[2]
        current_channels = 1 if preprocessed_data.ndim == 2 else preprocessed_data.shape[2]
        if current_channels < required_channels:
            preprocessed_data = np.repeat(preprocessed_data[..., np.newaxis], required_channels, -1)
        elif current_channels > required_channels:
            preprocessed_data = preprocessed_data[..., :required_channels]

    # Adding the batch dimension to the input data
    input_shape = (-1,) + preprocessed_data.shape

    # Reshape the input data
    preprocessed_data = np.reshape(preprocessed_data, input_shape)
    
    return preprocessed_data


def load_preprocessed_audio_data(file_path):
    # Load the .npy file
    preprocessed_data = np.load(file_path)

    return preprocessed_data

def upload_files():
    global uploaded_file_paths, listbox
    new_file_paths = filedialog.askopenfilenames()
    if len(new_file_paths) > 0:
        if len(uploaded_file_paths) + len(new_file_paths) > 500: #500 data uploadable files
            messagebox.showerror("Error", "Please select 500 or fewer audio files.")
            new_file_paths = ()
        else:
            uploaded_file_paths += new_file_paths
            for file_path in new_file_paths:
                listbox.insert(tk.END, basename(file_path))
            print(f"File paths: {uploaded_file_paths}")
            update_uploaded_files_label() 

def remove_selected_files():
    global uploaded_file_paths
    selected_indices = listbox.curselection()
    for index in reversed(selected_indices):
        # Remove the selected file from the listbox
        listbox.delete(index)

        # Remove the selected file from the global list
        del uploaded_file_paths[index]

    print(f"File paths: {uploaded_file_paths}")
    update_uploaded_files_label() 

def on_click(event):
    # Calculate the distance between the click event and all scatter points
    distances = np.linalg.norm(embeddings_2d[:, :2] - np.array([event.mouseevent.xdata, event.mouseevent.ydata]), axis=1)

    # Get the index of the closest point
    index = np.argmin(distances)

    file_name = basename(uploaded_file_paths[index])
    label = labels[index]
    messagebox.showinfo("Info", f"File: {file_name}\nLabel: {label}")
        

# Initialize the global file list
uploaded_file_paths = []

def run_model():
    global uploaded_file_paths, embeddings_2d, labels, embeddings

    if not uploaded_file_paths:
        messagebox.showerror("Error", "Please upload audio files first.")
        return
    
    if len(uploaded_file_paths) == 1:
        messagebox.showinfo("Info", "UMAP visualization requires at least two audio files. Please upload another file.")
        return

    embeddings = []
    labels = []

    try:
        for file_path in uploaded_file_paths:
            # Process the uploaded file and make a prediction using the model
            processed_data = predict_label(file_path, selected_model)  # Pass the selected_model variable
            predicted_probabilities = model.predict(processed_data)
            predicted_label_index = np.argmax(predicted_probabilities, axis=1)[0]
            predicted_label = label_mapping[int(predicted_label_index)]

            embeddings.append(np.squeeze(predicted_probabilities))  # Append predicted_probabilities instead of processed_data
            labels.append(predicted_label)

        # Display the predicted labels
        #label_text = "\n".join(f"File {i + 1}: {label}" for i, label in enumerate(labels))
        #messagebox.showinfo("Prediction", f"The predicted labels are:\n{label_text}")

        # Generate UMAP visualization
        #reducer = umap.UMAP(n_neighbors=25)
        reducer = umap.UMAP(n_components=2,n_neighbors=len(embeddings) - 1)
        embeddings = np.stack(embeddings, axis=0)
        embeddings_2d = reducer.fit_transform(embeddings)

        # Assign colors to the unique labels
        unique_labels = np.unique(labels)
        #cmap = plt.get_cmap('tab20', len(unique_labels))
        #cmap = plt.get_cmap('plasma', len(unique_labels))
        cmap = plt.get_cmap('viridis', len(unique_labels))
        label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

        # Create a Figure object for the UMAP plot
        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the UMAP visualization with different colors for each class label
        used_labels = set()
        for i, label in enumerate(labels):
            if label not in used_labels:
                ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=[label_colors[label]], label=label, picker=5)
                used_labels.add(label)
            else:
                ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=[label_colors[label]], picker=5)

        # Connect the onpick event to the UMAP plot
        fig.canvas.mpl_connect('pick_event', on_click)

        ax.set_title("UMAP Visualization of Audio Embeddings")
        ax.set_xlabel("UMAP Dimension 1") 
        ax.set_ylabel("UMAP Dimension 2")  

        # Creates a legend
        handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=label) for label, c in label_colors.items()]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        # Sets margins to leave space for the legend
        fig.subplots_adjust(right=0.8)

        # Create a new Toplevel window for the UMAP plot
        plot_window = tk.Toplevel(root)
        plot_window.title(f"UMAP - {selected_model.get()}")  # Include the currently selected model in the title
        plot_window.geometry("1600x800")
        plot_window.resizable(True, True)


        # Create a canvas widget to display the figure in the new window
        canvas = mpl_backend.FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.mpl_connect('pick_event', on_click)

        # Add a toolbar for navigating the UMAP plot
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="w")  # Replace pack with grid
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")  # Replace pack with grid

        # Configure the weights of the rows and columns
        plot_window.grid_rowconfigure(1, weight=1)
        plot_window.grid_columnconfigure(0, weight=1)


    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the audio files:\n{str(e)}")


# Create the GUI
root = tk.Tk()
root.title("Audio Classifier")
root.geometry("900x700")  
root.configure(bg="#2c3e50")

title_label = tk.Label(root, text="Audio Classifier", bg="#2c3e50", fg="#ecf0f1", font=("Helvetica", 20, "bold"))
title_label.pack(padx=10, pady=20)

# Model selection
model_names = ["Model 1", "Model 2", "Model 3"]
selected_model = tk.StringVar(root)
selected_model.set(model_names[0])  # Set the default, model 1
model = load_selected_model(selected_model.get())
selected_model.trace("w", model_selection_changed)

model_description_label = tk.Label(root, text=get_model_description(selected_model.get()), bg="#2c3e50", fg="#ecf0f1", font=("Helvetica", 12), wraplength=400, justify="left")
model_description_label.pack(padx=10, pady=10)

model_selection_label = tk.Label(root, text="Select Model:", bg="#2c3e50", fg="#ecf0f1", font=("Helvetica", 12, "bold"))
model_selection_label.pack(padx=10, pady=10)

model_selection_combobox = ttk.Combobox(root, textvariable=selected_model, values=model_names, state="readonly", font=("Helvetica", 12))
model_selection_combobox.pack(padx=10, pady=10)

upload_button = tk.Button(root, text="Upload Audio Files", command=upload_files, bg="#3498db", fg="#ecf0f1", font=("Helvetica", 12, "bold"), relief="groove")
upload_button.pack(padx=10, pady=10)

global uploaded_files_label
uploaded_files_label = tk.Label(root, text=f"Uploaded files: {len(uploaded_file_paths)}", bg="#2c3e50", fg="#ecf0f1", font=("Helvetica", 12))
uploaded_files_label.pack(padx=10, pady=10)


listbox_frame = tk.Frame(root)
listbox_frame.pack(padx=10, pady=10)

scrollbar = tk.Scrollbar(listbox_frame, orient="vertical")
scrollbar.pack(side="right", fill="y")

listbox = tk.Listbox(listbox_frame, bg="#ecf0f1", font=("Helvetica", 10), height=15, width=75, yscrollcommand=scrollbar.set)
listbox.pack(side="left", fill="both", expand=True)

listbox.config(yscrollcommand=scrollbar.set)

scrollbar.config(command=listbox.yview)

run_button = tk.Button(root, text="Run Model", command=run_model, bg="#27ae60", fg="#ecf0f1", font=("Helvetica", 12, "bold"), relief="groove")
run_button.pack(padx=10, pady=5)

remove_button = tk.Button(root, text="Remove Selected File", command=remove_selected_files, bg="#e74c3c", fg="#ecf0f1", font=("Helvetica", 12, "bold"), relief="groove")
remove_button.pack(padx=10, pady=10)

root.mainloop()