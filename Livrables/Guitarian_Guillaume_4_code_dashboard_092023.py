import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import tensorflow as tf
from tensorflow import keras
import tensorflow_advanced_segmentation_models as tasm
import cv2
import pickle

from collections import namedtuple


# Fixer la largeur de la page
# La fonction set_page_config devrait être compatible avec la version 1.15
st.set_page_config(layout="wide")


# Taille de Texte réajustable
default_font_size = 16
# Initialisation de la taille de la police si elle n'existe pas déjà dans le session_state
if 'font_size' not in st.session_state:
    st.session_state.font_size = default_font_size

# Fonction pour augmenter la taille de la police
def augmenter_police():
    st.session_state.font_size += 1 if st.session_state.font_size < 36 else 0

# Fonction pour diminuer la taille de la police
def diminuer_police():
    st.session_state.font_size -= 1 if st.session_state.font_size > 10 else 0

# Fonction pour RESET
def reset_default():
    st.session_state.font_size = default_font_size

# Deux boutons pour contrôler la taille de la police
col1, col2, col3, col4, col5, col6 ,col7, col8 ,col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23 = st.columns(23)
with col21:
    st.button("A+", on_click=augmenter_police)

with col22:
    st.button("A-", on_click=diminuer_police)

with col23:
    st.button("0", on_click=reset_default)



# Init des variables

static_url = "http://localhost:8501/app/static/"

# Création du dictionnaires des infos de dataset
dataset_dict = {
    'Images et Masque : ORIGINE': {
        'quantity': "5 000",
        'size': "2048 * 1024",
        'labels': 30,
        #'image': static_url + "image_orig.png",
        #'mask': static_url + "masque_orig.png"
    },'Images et Masque : REWORK': {
        'quantity': "3 475",
        'size': "128 * 128",
        'labels': 8,
        #'image': static_url + "image_rework.png",
        #'mask': static_url + "masque_rework.png"
    }
}
dataset_df = pd.DataFrame.from_dict(dataset_dict, orient='index')

# Création du dictionnaires des infos des résultats
score_dict = {
    'Baseline (UNet)': {
        'Train time (min)': 30,
        'Cat. Acc.': "0.849978",
        'mIoU': "0.575731",
        'loss': "0.537702",
        'epoch': 16
    },'HRNet (v2)': {
        'Train time (min)': 162,
        'Cat. Acc.': "0.893256",
        'mIoU': "0.630250",
        'loss': "0.390830",
        'epoch': 22
    }
}
score_df = pd.DataFrame.from_dict(score_dict, orient='index')


# Create tuple subclasses with named fields
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name the 30 classes

    'id'          , # An integer ID that is associated with this label.

    'categoryId'  , # IDs of labels which match to pixel values
                    # corresponding to the 8 main categories.

    'category'    , # The name of the category that this label belongs to
] )

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

labels = [
    #name                             id    catId   category
    Label(  'unlabeled'            ,  0 ,      0 , 'void'),
    Label(  'ego vehicle'          ,  1 ,      0 , 'void'),
    Label(  'rectification border' ,  2 ,      0 , 'void'),
    Label(  'out of roi'           ,  3 ,      0 , 'void'),
    Label(  'static'               ,  4 ,      0 , 'void'),
    Label(  'dynamic'              ,  5 ,      0 , 'void'),
    Label(  'ground'               ,  6 ,      0 , 'void'),
    Label(  'road'                 ,  7 ,      1 , 'flat'),
    Label(  'sidewalk'             ,  8 ,      1 , 'flat'),
    Label(  'parking'              ,  9 ,      1 , 'flat'),
    Label(  'rail track'           , 10 ,      1 , 'flat'),
    Label(  'building'             , 11 ,      2 , 'construction'),
    Label(  'wall'                 , 12 ,      2 , 'construction'),
    Label(  'fence'                , 13 ,      2 , 'construction'),
    Label(  'guard rail'           , 14 ,      2 , 'construction'),
    Label(  'bridge'               , 15 ,      2 , 'construction'),
    Label(  'tunnel'               , 16 ,      2 , 'construction'),
    Label(  'pole'                 , 17 ,      3 , 'object'),
    Label(  'polegroup'            , 18 ,      3 , 'object'),
    Label(  'traffic light'        , 19 ,      3 , 'object'),
    Label(  'traffic sign'         , 20 ,      3 , 'object'),
    Label(  'vegetation'           , 21 ,      4 , 'nature'),
    Label(  'terrain'              , 22 ,      4 , 'nature'),
    Label(  'sky'                  , 23 ,      5 , 'sky'),
    Label(  'person'               , 24 ,      6 , 'human'),
    Label(  'rider'                , 25 ,      6 , 'human'),
    Label(  'car'                  , 26 ,      7 , 'vehicle'),
    Label(  'truck'                , 27 ,      7 , 'vehicle'),
    Label(  'bus'                  , 28 ,      7 , 'vehicle'),
    Label(  'caravan'              , 29 ,      7 , 'vehicle'),
    Label(  'trailer'              , 30 ,      7 , 'vehicle'),
    Label(  'train'                , 31 ,      7 , 'vehicle'),
    Label(  'motorcycle'           , 32 ,      7 , 'vehicle'),
    Label(  'bicycle'              , 33 ,      7 , 'vehicle'),
    Label(  'license plate'        , -1 ,      7 , 'vehicle')]

# Création d'un dictionnaire de mapping
mapped_cat_dict = {label.id: label.categoryId for label in labels}
#La fonction de numpy appellé "vectorize" qui prend en entrée une fonction, nous permet de le faire.
mapped_cat_vect = np.vectorize(mapped_cat_dict.get)

# Créer un tableau de mapping de couleurs pour pouvoir afficher les MASQUES noirs et blanc en couleur
color_map = np.array([
    [249, 229, 196],    # 0 -> Jaune clair : void
    [192, 192, 192],    # 1 -> Gris clair : flat
    [208, 167, 30],     # 2 -> Or : construction
    [232, 86, 21],      # 3 -> Orange : object de la route
    [69, 136, 27],      # 4 -> Vert foncé : nature
    [96, 188, 255],     # 5 -> Bleu clair : sky
    [222, 66, 201],     # 6 -> Rose : human
    [0, 0, 0]           # 7 -> Noir : vehicle
], dtype=np.uint8)


img_height = 128
img_width = 128
img_size = (img_height, img_width)
img_size_cv2 = (img_width, img_height)
num_classes = 8
activation = 'sigmoid' if num_classes == 1 else 'softmax'
optim="adam"
# For Sparse y_true (mask file)
mean_IoU_sparse = tf.keras.metrics.MeanIoU(num_classes=num_classes, 
                                            name="mean_IoU",
                                            sparse_y_true=True, 
                                            sparse_y_pred=False)
# For Sparse y_true (mask file)
metrics_sparse = ['sparse_categorical_accuracy', mean_IoU_sparse]
metrics = metrics_sparse
loss = "sparse_categorical_crossentropy"

current_path = os.path.abspath(os.getcwd())
path_sep = os.path.sep
dest_masks_train_path = os.path.join(current_path, 'annotations', '8', 'train')

# Ouverture du fichier contenant les données du graphe de la distribution des pixels par classe
f = open('pixel_distrib_train.pckl', 'rb')
pixel_distrib_train = pickle.load(f)

image_dir = current_path + path_sep + 'app_images'
mask30_dir = current_path + path_sep + 'app_annotations'

reset_choice_label = 'reset choice'

image_list = os.listdir(image_dir)
image_list.insert(0, reset_choice_label)




##################################################################
################## DEBUT DE LA MISE EN PASGE #####################
##################################################################

# Titre du dashboard
st.markdown(f"<h1 style='font-weight: bold; font-size: {st.session_state.font_size * 3.5}px; text-align: center;'>Dashboard</h1>", unsafe_allow_html=True)


# Les onglets 'tabs' sont disponibles à partir de la version 0.84 de Streamlit.
tab1, tab2 = st.tabs(["DATASET", "POC"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        #st.write("### Dataset en quelques chiffres")
        st.markdown(f'<p style="font-weight: bold; font-size: {st.session_state.font_size * 1.5}px;">Dataset en quelques chiffres</p>', unsafe_allow_html=True)
        # La fonction st.dataframe est compatible avec Streamlit 1.15 pour afficher des DataFrames
        st.dataframe(dataset_df)

        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        # st.write("### Score des modèles baseline et HRNet")
        st.markdown(f'<p style="font-weight: bold; font-size: {st.session_state.font_size * 1.5}px;">Score des modèles baseline et HRNet</p>', unsafe_allow_html=True)
        st.dataframe(score_df)

    with col2:
        #st.write("### Distribution des classes de pixel dans les masques")
        st.markdown(f'<p style="font-weight: bold; font-size: {st.session_state.font_size * 1.5}px; text-align: center;">Distribution des classes de pixel dans les masques</p>', unsafe_allow_html=True)
        labels_data = {
            "final_counter": pixel_distrib_train['final_counter'],
            "categ": pixel_distrib_train['categ'],
            "color_map": pixel_distrib_train['color_map']
        }

        df = pd.DataFrame(labels_data)
        df['final_counter'] = df['final_counter'].astype(int)

        c = alt.Chart(df).mark_bar().encode(
            x=alt.X('categ', axis=alt.Axis(title='Les 8 catégories de pixels', labelAngle=-45)),
            y=alt.X('final_counter', axis=alt.Axis(title='QUANTITÉ', labelAngle=-45)),
            color=alt.Color('categ', scale=alt.Scale(range=labels_data['color_map']))
        ).interactive()
        st.altair_chart(c, use_container_width=True)

with tab2:

    @st.experimental_memo  # MISE EN CACHE : Pour que la méthode ne soit pas rééxecutée à chaque changement de police
    def getModelPrediction(image_rebuilt, modelType='HRNet'):

        if modelType == 'baseline':
            # Build model
            base_model, layers, layer_names = tasm.create_base_model(name='mobilenet',  height=img_height, width=img_width)
            model_UNet = tasm.UNet(n_classes=num_classes, base_model=base_model, output_layers=layers, backbone_trainable=True, final_activation=activation).model()
            model_UNet.compile(optimizer=optim, loss=loss, metrics=metrics)
            model_UNet.built = True
            model_UNet.load_weights("model_Unet2_mobilenet_sparse.h5") #Cette version ne fonctionne qu'avec les poids
            prediction = model_UNet.predict(image_rebuilt)

        else:
            # Build model
            HRNetOCR = tasm.HRNetOCR(n_classes=num_classes, height=img_height, width=img_width, final_activation=activation).model()
            HRNetOCR.compile(optimizer=optim, loss=loss, metrics=metrics)
            HRNetOCR.built = True
            HRNetOCR.load_weights("model_HRNetOCR_none_sparse.h5") #Cette version ne fonctionne qu'avec les poids
            prediction = HRNetOCR.predict(image_rebuilt)

        return prediction


    def get_prediction(selected_file, modelType='HRNet'):
        st.write('')
        st.write('')
        st.write('')

        if selected_file == reset_choice_label:
            #st.write('Please select an image name before trying to get prediction')
            st.markdown(f'<p style="font-size: {st.session_state.font_size * 1}px; text-align: center;">Veuillez selectionner une image !</p>', unsafe_allow_html=True)
        else:
            # Get the prediction
            # Preparing image to put into the models

            #resizing the image
            imageBGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_resize = cv2.resize(imageBGR, dsize=img_size_cv2)
            # Compatibility with model understanding
            imageFL32 = im_resize.astype(np.float32)
            # NORMALISATION
            image_ready = imageFL32/255

            # NOW we have to fit batch Format [1, image_width, image_height, number_of_channels]
            # in order to fit model imput
            image_batched_1 = np.expand_dims(image_ready, axis=0)

            # La Preparation des models se fait ici car sinon c'est toute la page
            # qui rame au changement d'image dans le selecteur 
            # Free up RAM in case the model definition cells were run multiple times
            keras.backend.clear_session()
            
            if modelType == 'baseline':
                #st.write('Unet')
                #st.markdown("<h5 style='text-align: center;'>Baseline (Unet)</h5>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-weight: bold; font-size: {st.session_state.font_size * 1.2}px; text-align: center;'>Baseline (Unet)</p>", unsafe_allow_html=True)
            else:
                #st.write('HRNet')
                #st.markdown("<h5 style='text-align: center;'>HRNet</h5>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-weight: bold; font-size: {st.session_state.font_size * 1.2}px; text-align: center;'>HRNet</p>", unsafe_allow_html=True)
                

            prediction = getModelPrediction(image_batched_1, modelType)
            predicted_mask = np.argmax(prediction[0], axis=-1)

            # Appliquer le mapping en utilisant l'indexation avancée de NumPy
            # Les dimensions sont ajustées pour que chaque pixel sélectionne la couleur correspondante
            predicted_mask_color = color_map[predicted_mask]

            #st.write(predicted_mask)
            #_, im_arr = cv2.imencode('.png', predicted_mask.astype("uint8"))  # im_arr: image in Numpy one-dim array format.
            predicted_mask = predicted_mask/np.max(predicted_mask)
            predicted_mask = (predicted_mask * 255).astype(np.uint8)
            # Displaying the masque
            #st.image(predicted_mask, caption=F"Predicted Mask (8 categ) {predicted_mask.shape}", use_column_width=True)

            # Afficher le mask prédit en couleur
            #st.image(predicted_mask_color, channels="RGB", caption=F"Predicted Mask (8 categ) {predicted_mask.shape}", use_column_width=True)
            st.image(predicted_mask_color, channels="RGB", use_column_width=True)
            st.markdown(f"<p style='font-weight: bold; font-size: {st.session_state.font_size * 1.2}px; text-align: center;'>Masque prédit (8 Cat / 128*128)</p>", unsafe_allow_html=True)


    col1, col2, col3, col4  = st.columns(4)


    with col1:
        #st.markdown(
        #"""
        ### Please, choose a file !
        #"""
        #)
        st.markdown(f"<p style='font-weight: bold; font-size: {st.session_state.font_size * 1.5}px; text-align: center;'>Selecteur d'image</p>", unsafe_allow_html=True)
        selected_file_name = st.selectbox(
            'empty',
            image_list,
            label_visibility="hidden"
        )

        if selected_file_name != reset_choice_label:
            ####### IMAGE #######
            st.markdown(f"<p style='font-weight: bold; font-size: {st.session_state.font_size * 1}px; text-align: center;'>Image réelle d'origine</p>", unsafe_allow_html=True)
            image_fn_path = image_dir+path_sep+selected_file_name
            mask_fn_path = mask30_dir+path_sep+selected_file_name
            image = cv2.imread(image_dir+path_sep+selected_file_name)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #st.image(image, caption='Real Image', channels = 'BGR')
            st.image(image, channels = 'BGR')
        else :
            image = None
            mask = None

    
    with col2:
        #st.header("A dog")
        if selected_file_name != reset_choice_label:
            
            ####### MASK #######
            mask_30categ = cv2.imread(mask30_dir+path_sep+selected_file_name, cv2.IMREAD_GRAYSCALE)
            # Convert it via the mapper into 8 categ matrix
            mask = mapped_cat_vect(mask_30categ)
            #resizing the mask
            mask = cv2.resize(mask, dsize=img_size_cv2, interpolation=cv2.INTER_NEAREST)
            #st.write(mask)

            # Appliquer le mapping en utilisant l'indexation avancée de NumPy
            # Les dimensions sont ajustées pour que chaque pixel sélectionne la couleur correspondante
            mask_color = color_map[mask]

            #better see grayscale image
            mask = mask/np.max(mask)
            mask = (mask * 255).astype(np.uint8)

            st.write('')
            st.write('')
            st.write('')
            #st.write('Mask GT')
            #st.markdown("<h5 style='text-align: center;'>Mask GT</h5>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-weight: bold; font-size: {st.session_state.font_size * 1.2}px; text-align: center;'>Masque vérité terrain (8 Cat / 128*128)</p>", unsafe_allow_html=True)

            # Afficher le mask en noir et blanc
            #st.image(mask, caption=F"GT Mask converted to 8 categories and resized to {mask.shape}", use_column_width=True)

            # Afficher le mask en couleur
            # st.image(mask_color, channels="RGB", caption=F"GT Mask converted to 8 categories and resized to {mask.shape}", use_column_width=True)
            st.image(mask_color, channels="RGB", use_column_width=True)
            
        else :
            image = None
            mask = None
    
    
    with col3:        
        get_prediction(selected_file_name, 'baseline')

    with col4:        
        get_prediction(selected_file_name)