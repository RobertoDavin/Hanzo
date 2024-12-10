import sys
import os
import cv2
import onnxruntime as ort
import cv2
import numpy as np
import torch
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# sys.path.append("Monk_Object_Detection/4_efficientdet/lib/")

class_labels = ["mandarin",   "1 ban 4",   "1 jie 2",   "1 shu 4",   "1 tong 2",   "1 wei 4", 
                        "1 xin 1",    "1 yi 4",    "1 yue 4",    "1 zhi 1",    "2 bao 4",    
                        "2 fu 2",    "2 jian 4",    "2 jie 2",    "2 qu 2",    "2 shu 4",    
                        "2 ti 4",    "2 zhi 1",    "3 bao 4",    "3 fu 2",    "3 heng 2",    
                        "3 jian 4",    "3 meng 2",    "3 mo 2",    "3 ti 4",    "3 xin 1",    
                        "3 yi 4",    "4 ban 4",    "4 heng 2",    "4 meng 2",    "4 mo 2",    
                        "4 qu 2",    "4 tong 2",    "4 wei 4",    "4 yue 4",    "an 4",    
                        "ao 2",    "ba 1",    "ban 3",    "bao 2",    "bei 1",    
                        "ben 3",    "bi 4",    "bo 4",    "bu 4",    "cai 2",    
                        "cang 2",    "cao 1",    "cao 2",    "ceng 2",    "chan 4",    
                        "chang 3",    "chao 2",    "cheng 2",    "chu 2",    "chun 1",    
                        "chun 3",    "cong 2",    "cu 4",    "cui 4",    "cun 1",    
                        "dai 3",    "dai 4",    "dan 4",    "dian 1",    "ding 1",    
                        "du 4",    "duo 3",    "er 4",    "fa 2",    "fan 2",    
                        "fu 4",    "ge 1",    "ge 4",    "geng 4",    "gong 1",    
                        "gou 4",    "guan 4",    "guo 3",    "hang 2",    "hao 2",    
                        "he 2",    "he 4",    "hui 1",    "hui 4",    "ji 1",    
                        "ji 2",    "jia 4",    "jiang 1",    "jiao 1",    "jiao 4",    
                        "jie 3",    "jin 3",    "jin 4",    "jiu 3",    "ju 1",    
                        "kang 4",    "kou 3",    "ku 1",    "kui 4",    "lai 2",    
                        "lan 3",    "lang 3",    "li 3",    "li 4",    "lian 4",    
                        "lin 2",    "lin 3",    "luo 3",    "man 4",    "mao 1",    
                        "mei 3",    "men",    "mi 2",    "ming 2",    "mo 4",    
                        "mu 3",    "mu 4",    "nang 3",    "nuo 4",    "pan 2",    
                        "pei 2",    "peng 2",    "pu 2",    "pu 3",    "qi 1",    
                        "qian 4",    "qiang 1",    "qiang 2",    "qiao 1",    "qiao 2",    
                        "qiao 4",    "qing 2",    "qu 1",    "qu 4",    "rang 2",    
                        "rao 3",    "ren 2",    "ri 4",    "ru 4",    "san 1",    
                        "sha 1",    "sha 4",    "shan 1",    "she 4",    "shu 1",    
                        "shu 3",    "shui 3",    "si 3",    "song 1",    "tan 3",    
                        "tang 2",    "teng 2",    "tiao 2",    "tu 3",    "wai 1",    
                        "wang 2",    "wang 4",    "wei 1",    "wen 1",    "wu 3",    
                        "xi 1",    "xi 2",    "xian 3",    "xiang 1",    "xiang 4",    
                        "xiao 1",    "xiao 3",    "xie 1",    "xie 4",    "xiong 1",    
                        "xiu 3",    "xu 1",    "xun 1",    "ya 1",    "yang 2",    
                        "yao 4",    "ye 4",    "yi 1",    "ying 1",    "you 3",    
                        "you 4",    "yu 3",    "yu 4",    "yuan 2",    "zan 4",    
                        "zha 2",    "zhang 1",    "zhang 4",    "zhao 4",    "zhe 4",    
                        "zhen 3",    "zhen 4",    "zheng 4",    "zhi 3",    "zhi 4",    
                        "zhu 1",    "zong 1",    "zui 4",    "zun 1",    "zuo 3"]

class_labels2 = ["mandarin", "ni 3", "ta 1", "wo 3"]

def mapping_pinyin(pinyin):
    group_mapping = {
        "Group 1 (1 - 5 Strokes)": [
            "1 ban 4", "1 jie 2", "1 shu 4", "1 tong 2", "1 wei 4", "1 xin 1", "1 yi 4", "1 yue 4", "1 zhi 1",
            "ba 1", "ben 3", "chang 3", "cong 2", "dai 3", "ding 1", "er 4", "fa 2", "ge 4", "gong 1", "jin 3",
            "jiu 3", "kang 4", "kou 3", "li 4", "men", "mo 4", "mu 3", "mu 4", "pu 2", "qu 4", "ren 2", "ri 4",
            "ru 4", "san 1", "shu 1", "shui 3", "tu 3", "wang 2", "xi 2", "xiao 3", "xiong 1", "ya 1", "yi 1",
            "you 4", "yu 3", "yuan 2", "zha 2", "zheng 4", "zhi 3", "zuo 3", "ta 1"
        ], # Group 1 (1-5)
        "Group 2 (6 - 10 Strokes)": [
            "2 bao 4", "2 fu 2", "2 jian 4", "2 jie 2", "2 qu 2", "2 shu 4", "2 ti 4", "2 zhi 1", "ban 3", "bei 1",
            "bi 4", "bu 4", "cai 2", "cun 1", "du 4", "duo 3", "geng 4", "gou 4", "guo 3", "hang 2", "he 2",
            "ji 1", "jia 4", "ku 1", "lai 2", "lang 3", "li 3", "mei 3", "ming 2", "peng 2", "pu 3", "qiang 1",
            "qu 1", "rao 3", "sha 1", "shan 1", "si 3", "song 1", "tiao 2", "wai 1", "wu 3", "xiu 3", "yang 2", 
            "ye 4", "you 3", "zhang 4", "zhao 4", "zhen 3", "zhen 4", "zhu 1", "ni 3", "wo 3"
        ], # Group 2 (6-10)
        "Group 3 (11 - 15 Strokes)": [
            "3 bao 4", "3 fu 2", "3 heng 2", "3 jian 4", "3 meng 2", "3 mo 2", "3 ti 4", "3 xin 1", "3 yi 4",
            "cao 2", "ceng 2", "chao 2", "chun 1", "cui 4", "dai 4", "dan 4", "fan 2", "ge 1", "hao 2", "he 4",
            "jiao 1", "jiao 4", "jie 3", "jin 4", "ju 1", "kui 4", "lian 4", "luo 3", "man 4", "mao 1", "pei 2",
            "qi 1", "qian 4", "qiang 2", "qing 2", "sha 4", "tan 3", "tang 2", "wang 4", "wei 1", "wen 1", "xiang 4",
            "xie 1", "xu 1", "ying 1", "yu 4", "zhang 1", "zhe 4", "zhi 4", "zui 4"
        ], # Group 3 (11-15)
        "Group 4 (16 - 20 Strokes)": [
            "4 ban 4", "4 heng 2", "4 meng 2", "4 mo 2", "4 qu 2", "4 tong 2", "4 wei 4", "4 yue 4", "an 4", "ao 2",
            "bao 2", "bo 4", "cang 2", "cao 1", "chan 4", "cheng 2", "chu 2", "chun 3", "cu 4", "dian 1", "fu 4",
            "guan 4", "hui 1", "hui 4", "ji 2", "jiang 1", "lan 3", "lin 2", "lin 3", "mi 2", "nang 3", "nuo 4",
            "pan 2", "qiao 1", "qiao 2", "qiao 4", "rang 2", "she 4", "shu 3", "teng 2", "xi 1", "xian 3", "xiang 1",
            "xiao 1", "xie 4", "xun 1", "yao 4", "zan 4", "zong 1", "zun 1"
        ] # Group 4 (16-20)
    }
    
    for group, pinyin_list in group_mapping.items():
        if pinyin in pinyin_list:
            return group
    return "Unknown"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Home Page class definition
class HomePage(QWidget):
    def __init__(self, stacked_widget):
        self.model = 1
        super().__init__()
        self.stacked_widget = stacked_widget  # Reference to the stacked widget for page switching

        self.setWindowTitle('Home')  # Window title
        self.setGeometry(100, 100, 600, 400)  # Window size

        main_layout = QVBoxLayout()  # Main layout for the Home page
        title_label = QLabel("Hanzo\n"
                             "'Hanzi and Go!'", self)
        title_label.setAlignment(Qt.AlignCenter)  # Center the title label
        title_label.setStyleSheet("font-size: 40px; font-weight: bold;")  # Title style

        # Start Recognition button
        start_button = QPushButton("Start Recognition", self)
        start_button.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                padding: 15px;
                background-color: #e0f0ff;
                border: 2px solid #c0d0ff;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #d0e0ff;
            }
        """)  # Button styling
        start_button.setFixedSize(250, 75)
        start_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        start_button.clicked.connect(self.go_to_recognition_page)  # Connect button to page switch

        # Create About and Help buttons (top-right)
        top_buttons_layout = QHBoxLayout()
        about_button = QPushButton("About", self)
        about_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        help_button = QPushButton("Help", self)
        help_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering

        # Styling for About and Help buttons
        for btn in [about_button, help_button]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 8px 20px;
                    border: 1px solid #888;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
            """)
            btn.setFixedSize(100, 40)

        # Add About and Help buttons to the layout and align them to the right
        top_buttons_layout.addStretch()
        top_buttons_layout.addWidget(about_button)
        top_buttons_layout.addWidget(help_button)
        
        # Add functionality to the Help & About button to switch to Help & About page
        help_button.clicked.connect(self.go_to_help_page)
        about_button.clicked.connect(self.go_to_about_page)
        
        # Add elements to the main layout
        main_layout.addLayout(top_buttons_layout)
        main_layout.addStretch()  # Adds spacing between elements
        main_layout.addWidget(title_label)
        main_layout.addSpacing(20)
        main_layout.addWidget(start_button, alignment=Qt.AlignCenter)
        main_layout.addStretch()

        self.setLayout(main_layout)  # Set the final layout

    # Method to switch to the Recognition Page
    def go_to_recognition_page(self):
        self.stacked_widget.setCurrentIndex(1)
        
    # Method to switch to the Help Page
    def go_to_help_page(self):
        self.stacked_widget.setCurrentIndex(3)
    
    # Method to switch to the About Page
    def go_to_about_page(self):
        self.stacked_widget.setCurrentIndex(4)
        
# Recognition Page class definition
class RecognitionPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.uploaded_image_path = None  # To store the uploaded image path
        self.model_type = 1
        self.setWindowTitle('Recognition')
        self.setGeometry(100, 100, 600, 400)

        main_layout = QVBoxLayout()
        back_button = QPushButton("Back", self)
        back_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        back_button.setFixedSize(90, 40)
        back_button.setStyleSheet("font-size: 14px;")
        back_button.clicked.connect(self.go_back_home)  # Connect back button to Home page
        
        # Placeholder for the uploaded image
        self.image_placeholder = QLabel("", self)
        self.image_placeholder.setFixedSize(512, 512)
        self.image_placeholder.setAlignment(Qt.AlignCenter)
        self.image_placeholder.setStyleSheet("border: 1px solid black;")

        # Buttons at the bottom: Upload Photo and Send
        button_layout = QHBoxLayout()
        upload_button = QPushButton("Upload Image", self)
        upload_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        send_button = QPushButton("Send", self)
        send_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        add_data_button = QPushButton("Add Character", self)  # New Add Data button
        add_data_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering

        # Styling for the buttons
        for btn in [upload_button, send_button, add_data_button]:
            btn.setFixedSize(160, 60)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    background-color: #e0f0ff;
                    border: 2px solid #c0d0ff;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #d0e0ff;
                }
            """)

        # Connect the buttons to their functions
        upload_button.clicked.connect(self.open_file_dialog)  # Opens file dialog to upload image
        send_button.clicked.connect(self.go_to_result_page)  # Switches to Result Page
        add_data_button.clicked.connect(self.add_data_functionality)  # Opens Add Data functionality

        button_layout.addStretch()  # Align buttons at the center
        button_layout.addWidget(upload_button)
        button_layout.addWidget(send_button)
        button_layout.addWidget(add_data_button)  # Add Data button added to layout
        button_layout.addStretch()

        # Add elements to the main layout
        main_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(self.image_placeholder, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    # Opens file dialog to select an image and displays it
    def open_file_dialog(self):
        self.uploaded_image_path = None
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *jpeg *.bmp)", options=options)
        if file_name:  # If a file is selected
            self.uploaded_image_path = file_name  # Store the file path
            pixmap = QPixmap(file_name)  # Load the image
            self.image_placeholder.setPixmap(pixmap.scaled(self.image_placeholder.size(), Qt.KeepAspectRatio))  # Display the image

    # Switches to the Result Page
    def go_to_result_page(self):
        try:
            if os.path.isfile(self.uploaded_image_path):
                predicted_label = self.predict_image(self.uploaded_image_path)
                predicted_group = mapping_pinyin(predicted_label)
                #predicted_label = predicted_label + ", \n Predicted Group: " + mapping_pinyin(predicted_label)
                self.stacked_widget.widget(2).display_uploaded_image(self.uploaded_image_path, predicted_label, predicted_group)  # Pass the image to the Result Page
                self.stacked_widget.setCurrentIndex(2)
        except:
            pass

    # Clears the image placeholder (resets to blank state)
    def clear_image(self):
        self.image_placeholder.clear()  # Clear the pixmap
        self.image_placeholder.setText("")  # Reset the label text

    # Switch back to the Home page
    def go_back_home(self):
        self.uploaded_image_path = None
        if os.path.exists('tmp.png'):
            os.remove('tmp.png')
        self.clear_image()  # Clear the uploaded image when going back
        self.stacked_widget.setCurrentIndex(0)
    
    # Placeholder for Add Data functionality
    def add_data_functionality(self):
        self.model_type = 2
        file_name, _ = QFileDialog.getOpenFileName(self, "Select New Character Image", "", "Images (*.png *.jpg *jpeg *.bmp)")
        if file_name:
            text, ok = QInputDialog.getText(self, "Add Data", "Enter the Pinyin label for the character:")
            if ok and text.strip():
                # Save the image and Pinyin to the dataset directory
                new_label_dir = os.path.join("data", text.strip())
                os.makedirs(new_label_dir, exist_ok=True)
                shutil.copy(file_name, os.path.join(new_label_dir, os.path.basename(file_name)))
                QMessageBox.information(self, "Success", f"Data added successfully under label: {text.strip()}!")
            else:
                QMessageBox.warning(self, "Input Error", "You must enter a valid Pinyin label!")
    
    def predict_image(self, image_path):
        # Setting configurations
        system_dict = {}
        system_dict["verbose"] = 1
        system_dict["local"] = {}
        system_dict["local"]["common_size"] = 512
        system_dict["local"]["mean"] = np.array([[[0.485, 0.456, 0.406]]])
        system_dict["local"]["std"] = np.array([[[0.229, 0.224, 0.225]]])
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        image = img.astype(np.float32) / 255 # Normalization
        image = (image.astype(np.float32) - system_dict["local"]["mean"]) / system_dict["local"]["std"] # Apply mean and std normalization 
        height, width, _ = image.shape

        # Resize the image to a common size of 512x512
        # center-crop the image to a 512x512 region
        if height != width:
            start_row = int((height - 512) / 2)
            start_col = int((width - 512) / 2)
            image = image[start_row:start_row+512, start_col:start_col+512]
        # scale the dimensions accordingly and pad the resized image to make it square (512x512)
        if height > width:
            scale = system_dict["local"]["common_size"] / height
            resized_height = system_dict["local"]["common_size"]
            resized_width = int(width * scale)
        else:
            scale = system_dict["local"]["common_size"] / width
            resized_height = int(height * scale)
            resized_width = system_dict["local"]["common_size"]

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((system_dict["local"]["common_size"], system_dict["local"]["common_size"], 3)) # Creates a new image with the size of 512 x 512 x 3
        new_image[0:resized_height, 0:resized_width] = image

        # Converts the numpy array into a tensor, and adjust the dimensions
        img = torch.from_numpy(new_image)
        img_tensor = img.cpu().permute(2, 0, 1).float().unsqueeze(dim=0) # Channels, Height, Width
        img_float = img_tensor.numpy()
        
        # Import the weights of the model used
        if self.model_type == 1:
          model_path = "efficientdet.onnx"
          class_label_used = class_labels
        elif self.model_type == 2:
          model_path = "efficientdet_add.onnx"
          class_label_used = class_labels2
          
        session = ort.InferenceSession(resource_path(model_path))
        
        input_data = img_float.astype(np.float32)
        # Get the input and output names for the model
        input_name = session.get_inputs()[0].name
        score_name = session.get_outputs()[0].name
        label_name = session.get_outputs()[1].name
        box_name = session.get_outputs()[2].name

        # Run the model Inference
        # Predictions = [[[0.78], [0.32]], -> Score
        #               [[70], [3]], -> Label
        #               [[120, 210, 200, 310], [312, 423, 231, 453]] -> Bounding box
        scores, labels, boxes = session.run([score_name, label_name, box_name], {input_name: input_data})
        boxes /= scale
        
        try:
            if(len(scores) == 0):
                predicted_label = "None"
                return predicted_label
            
            image = cv2.imread(image_path)
            #for i in range(len(scores)):
            #    x_min, y_min, x_max, y_max = boxes[i]
            #    class_label = class_labels[labels[i]]
            #    score = scores[i]
                
            #    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2) # (RGB), Thickness
            #    label_text = f'{class_label}: {score:.2f}'
            #    cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #            1, (0, 0, 255), 2)
            
            x_min, y_min, x_max, y_max = boxes[0]
            class_label = class_label_used[labels[0]]
            score = scores[0]
            
            # Draw bounding box and label on the final_image
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2) # (RGB), Thickness
            label_text = f'{class_label}: {score:.2f}'
            cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            width, height, _ = image.shape
            if width != height:
                image = cv2.resize(image, (512, 512))
            image = cv2.resize(image, (512, 512))
            # final_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('tmp.png', image) # Saves the image with bounding box applied
            predicted_label = self.class_to_label(labels[0]) # Label Prediction
        except:
            predicted_label = "None"
        return predicted_label
      
    def class_to_label(self, class_idx):
        # Map the class index to the corresponding Mandarin character or Pinyin
        if self.model_type == 1:
          return class_labels[class_idx]
        else:
          return class_labels2[class_idx]

# Result Page class definition
class ResultPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.uploaded_image_path = None 

        self.setWindowTitle('Result')
        self.setGeometry(100, 100, 600, 400)

        main_layout = QVBoxLayout()
        back_button = QPushButton("Back", self)
        back_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        back_button.setFixedSize(90, 40)
        back_button.setStyleSheet("font-size: 14px;")
        back_button.clicked.connect(self.go_back_recognition)  # Back button to Recognition Page

        # Placeholder for the result image
        self.image_result = QLabel("", self)
        self.image_result.setFixedSize(512, 512)
        self.image_result.setAlignment(Qt.AlignCenter)
        self.image_result.setStyleSheet("border: 1px solid black;")

        # Placeholder for Pinyin result (for now it's just a static label)
        self.pinyin_result = QLabel("Hasil Romanisasi (Pinyin)", self)
        self.pinyin_result.setAlignment(Qt.AlignCenter)
        self.pinyin_result.setStyleSheet("font-size: 24px;")

        # Button to return to Home
        back_home_button = QPushButton("Back to Home", self)
        back_home_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        back_home_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 10px;
                background-color: #e0f0ff;
                border: 2px solid #c0d0ff;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #d0e0ff;
            }
        """)
        back_home_button.setFixedSize(160, 60)
        back_home_button.clicked.connect(self.go_back_home)  # Return to Home page

        # Add elements to the main layout
        main_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(self.image_result, alignment=Qt.AlignCenter)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.pinyin_result, alignment=Qt.AlignCenter)
        main_layout.addSpacing(20)
        main_layout.addWidget(back_home_button, alignment=Qt.AlignCenter)
        main_layout.addStretch()

        self.setLayout(main_layout)

    # Display the uploaded image in the result page
    def display_uploaded_image(self, image_path, predicted_label, predicted_group):
        if os.path.exists('tmp.png'):
            image_path = 'tmp.png'
        pixmap = QPixmap(image_path)
        self.image_result.setPixmap(pixmap.scaled(self.image_result.size(), Qt.KeepAspectRatio))  # Display the image
        # Display the predicted label (Pinyin)
        self.pinyin_result.setText(f"Predicted Pinyin: {predicted_label} \n Predicted Group: {predicted_group}")

    # Switch back to the Recognition page
    def go_back_recognition(self):
        recog_page = self.stacked_widget.widget(1)
        if recog_page.model_type == 2:
            recog_page.model_type = 1
        self.stacked_widget.widget(1).clear_image()  # Clear the image placeholder on the Recognition Page
        self.stacked_widget.setCurrentIndex(1)

    # Return to the Home page
    def go_back_home(self):
        recog_page = self.stacked_widget.widget(1)
        if recog_page.model_type == 2:
            recog_page.model_type = 1
        if os.path.exists('tmp.png'):
            os.remove('tmp.png')
        self.uploaded_image_path = None
        self.stacked_widget.widget(1).clear_image()  # Clear the image placeholder on the Recognition Page
        self.stacked_widget.setCurrentIndex(0)

# Help Page class definition
class HelpPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget  # Reference to the stacked widget
        self.uploaded_image_path = None 
        self.setWindowTitle('Help')  # Set window title
        self.setGeometry(100, 100, 600, 400)  # Set window size

        main_layout = QVBoxLayout()  # Main layout for the Help page

        # Back button to return to the Home page
        back_button = QPushButton("Back", self)
        back_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        back_button.setFixedSize(90, 40)
        back_button.setStyleSheet("font-size: 14px;")
        back_button.clicked.connect(self.go_back_home)  # Connect back button to go back home

        # Instructions title
        title_label = QLabel("Instructions\n", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 34px; font-weight: bold;")  # Styling for the title

        # Instruction content (just an example text for now)
        instructions_label = QLabel("To use Hanzo, please follow the steps below:\n\n"
                            "1. To start using Hanzo, click on 'Start Recognition'.\n"
                            "2. Click 'Upload Image' to upload an image of a Mandarin character.\n"
                            "3. Click 'Send' to get the predicted Pinyin for the character.\n"
                            "4. Click 'Add Charcater' to add a new character's picture and input the Pinyin for the character.\n"
                            "5. The recognition result will be shown below the image preview of the result page.\n"
                            "6. Click 'Back to Home' to return to the home page.\n"
                            "7. Use the 'Back' buttons to return to the previous pages.\n"
                            "8. Visit 'About' for more information about Hanzo.", self)
        #instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setStyleSheet("font-size: 22px;")  # Styling for the instruction text

        # Add elements to the main layout
        main_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(instructions_label, alignment=Qt.AlignCenter)
        main_layout.addStretch()

        self.setLayout(main_layout)  # Set the layout

    # Switch back to the Home page
    def go_back_home(self):
        self.uploaded_image_path = None
        if os.path.exists('tmp.png'):
            os.remove('tmp.png')
        self.stacked_widget.setCurrentIndex(0)

# About Page class definition
class AboutPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget  # Reference to the stacked widget
        self.uploaded_image_path = None 
        self.setWindowTitle('About')  # Set window title
        self.setGeometry(100, 100, 600, 400)  # Set window size

        main_layout = QVBoxLayout()  # Main layout for the Help page

        # Back button to return to the Home page
        back_button = QPushButton("Back", self)
        back_button.setCursor(Qt.PointingHandCursor)  # Change cursor when hovering
        back_button.setFixedSize(90, 40)
        back_button.setStyleSheet("font-size: 14px;")
        back_button.clicked.connect(self.go_back_home)  # Connect back button to go back home

        # About title
        title_label = QLabel("About\n", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 34px; font-weight: bold;")  # Styling for the title

        # About content (just an example text for now)
        about_label = QLabel("About Hanzo:\n\n"
                            "1. Hanzo - 'Hanzi and Go!' was developed as part of a final thesis project.\n"
                            "2. Its primary function is to recognize 200 Mandarin characters and provide the \nPinyin romanization as the output.\n"
                            "3. The developer of Hanzo is Roberto Davin, with the NIM 535210022.\n"
                            "4. Hanzo utilizes advanced machine learning techniques, specifically the \nEfficientDet model, for character detection and recognition.\n"
                            "5. Hanzo aims to contribute to advancements in Mandarin OCR (Optical \nCharacter Recognition) technology.\n", self)
        #about_label.setAlignment(Qt.AlignCenter)
        about_label.setStyleSheet("font-size: 22px;")  # Styling for the about text
        
        # Add elements to the main layout
        main_layout.addWidget(back_button, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(about_label, alignment=Qt.AlignCenter)
        main_layout.addStretch()

        self.setLayout(main_layout)  # Set the layout

    # Switch back to the Home page
    def go_back_home(self):
        self.uploaded_image_path = None
        if os.path.exists('tmp.png'):
            os.remove('tmp.png')
        self.stacked_widget.setCurrentIndex(0)

# Main Window class to manage pages
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.stacked_widget = QStackedWidget()

        # Create instances of each page
        self.home_page = HomePage(self.stacked_widget)
        self.recognition_page = RecognitionPage(self.stacked_widget)
        self.result_page = ResultPage(self.stacked_widget)
        self.help_page = HelpPage(self.stacked_widget)
        self.about_page = AboutPage(self.stacked_widget)

        # Add the pages to the stacked widget
        self.stacked_widget.addWidget(self.home_page)  # Index 0: Home Page
        self.stacked_widget.addWidget(self.recognition_page)  # Index 1: Recognition Page
        self.stacked_widget.addWidget(self.result_page)  # Index 2: Result Page
        self.stacked_widget.addWidget(self.help_page) # Index 3: Help Page
        self.stacked_widget.addWidget(self.about_page) # Index 4: About Page

        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)  # Add the stacked widget to the main layout
        self.setLayout(layout)

        self.setWindowTitle('Hanzo')  # Window title
        self.setGeometry(100, 100, 1200, 800)  # Window size

def delete_temp_image():
    temp_image_path = 'tmp.png'
    if os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except Exception as e:
            print(f"Error while deleting {temp_image_path}: {e}")

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()  # Initialize the main window
    main_window.show()  # Show the main window
    app.aboutToQuit.connect(delete_temp_image)
    sys.exit(app.exec_())  # Run the application loop
