import os
import torch
import random
import pandas as pd
import numpy as np

from cv2 import imread
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms, utils
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Sampler

class BrodenDataset(Dataset):
    
    """
    Args:
        csv_file (string): path to training_data.csv/test_data.csv of the Broden dataset
        transform (callabel, optional): Optional transform to be applied on the sample
    
    Returns: 
        dictionary
    """
    
    def __init__(self, csv_file, data_path, transform = None):
        
        self.temp_df = pd.read_csv(csv_file)
        self.data_path = data_path
        self.transform = transform
        self.rows, self.columns = self.temp_df.shape
        self.id_matrix = np.eye(self.columns) # create a one-hot vector for each class 
        
    def __len__(self):
        return len(self.temp_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        im = io.imread(os.path.join(self.data_path, 'images', self.temp_df.loc[idx, 'image']))
        
        if self.transform:
            im = self.transform(im)
        
        #ToDo: change this code, I do not know how to subset over columns based on row values in a dataframe
        #this code is messy but works
        b_array = self.temp_df.loc[idx, :].apply(lambda x: x == 1).values # create boolean array of the classes in the picture
        concept_names = self.temp_df.columns.values[b_array] #subset all column names based on the boolean array
        ##
        
        col_index = [self.temp_df.columns.get_loc(c) for c in concept_names if c in self.temp_df] # get the column index of the above defined classes
        concept_vectors = self.id_matrix[col_index] # get the one-hot vector for each class in the picture
        
        sample = {'image' : im, 'concept': concept_names, 'concept vectors': concept_vectors}
        
        return sample

def MakeVectorDictionary(nn, out_layer, dataset, index_values, file_name=None):
    """
    Creates a dictionary with the row indices as key and the output tensor as value. The CNN is set to evaluation mode
    after which every image is run through the CNN. The output tensor of the 'avgpool' layer is stored as value in the 
    dictionary. After every image has been passed through the network, the dictionary is saved on the local harddrive.
    
    Args:
        nn (torchvision.model): a CNN which has a 'avgpool' layer before the fully connected layers
        out_layer: a layer of the nn, from which the output will be extracted
        dataset (torch.utils.data.Dataset): a dataset made with the function 'BrodenDataset'
        index_values (list): a list of integers refering to the row indices of the images labelled with a specific concept
                            in the training dataframe
        file_name (.pickle): default is None, the dictionary will not be stored.
                            If specified, the name with which the dictionary will be stored on the local harddrive. 
                            It must end with '.pickle'. The file will be stored in the data folder
        
    
    Returns:
        dictionary
    """
    
    #check if a GPU is available, otherwise run it on CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nn.to(device)
    # set the neural network to evaluation mode
    nn.eval()
    
    # create a dictionary in which the tensors will be stored
    vector_dict = {}
    
    # loop through every index, push the image belonging to the index through the network and extract the tensor from
    # the desired output layer
    for idx in tqdm_notebook(index_values):
        try:
            sample = dataset[idx]
            img = sample['image']
            if torch.cuda.is_available():
                img = img.float().cuda()
            else:
                img = img.float()
            vector_img = GetVector(img, nn, out_layer)
            vector_dict[str(idx)] = vector_img
        except:
            print('index:', idx)
            
    # write the dictionary to the local harddrive
    if file_name is not None:
        with open(os.path.join('../data/', file_name), 'wb') as handle:
            pickle.dump(vector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return vector_dict


def GetCosineSimilarityDistance(image_index, img_vector_dict, similarity_value):
    
    """
    Calculates the cosine similarity distance between the tensors stored in the dictionary
    
    Args:
        image_index (string): the index of the image in the training dataframe which will be used as reference image
        img_vector_dict (dictionary): a dictionary with the row index of images as key and their belonging tensor as value
        similarity_value (float): ranging from 1 to -1, determines how similar the tensors must be. A value of 1 will return 
        1 tensor as only the reference tensor is equal to itself. A value of -1 will return all tensors. 
        
    Returns:
        index value of reference image (integer)
        index values of tensors which have a cosine similarity which is larger or equal the similarity value (list)
    """
    
    # raise an error if the similarity value does not range between 1 and -1
    if similarity_value >= 1 and similarity_value < -1:
        raise ValueError ('the similarity_value must range between 1 to -1, but got %f instead' %similarity_value)
    
    # get the tensor belonging the key and reshape it to the right dimensions
    reference_vector = img_vector_dict[image_index].unsqueeze(0)
    
    # create a list of all the keys in the dictionary
    list_of_keys = [key for key in img_vector_dict.keys()]
    
    # stack all the tensors 
    vector_stack = torch.stack([img_vector_dict[key] for key in list_of_keys])
    
    # calculate the cosine similarity between the reference vector and all other vectors
    cos_distance = F.cosine_similarity(vector_stack, reference_vector)
    
    # sort the cosine similarity
    cos_sim, index = cos_distance.sort(descending = True)
    
    # subset the images which fall within the range of the similarity value and convert the indices to a list
    img_key_indices = index[:len(cos_sim[cos_sim >= similarity_value])].tolist()
    
    # the indices represent the index of the tensor in the list of keys, these must be converted to the image indices
    img_indices = []
    for ix in img_key_indices:
        img_indices.append(list_of_keys[ix])
    
    return img_indices

def GetVector(img_tensor, net, output_layer):
    
    """
    The function extracts the feature vector of an image from the 'avgpool' layer in a model
    
    Args:
        img_tensor (torch.Tensor): a tensor of an image
        net (torchvision.model): a Neural Network or Convolutional NN
        out_layer: a layer of the nn, from which the output will be extracted
    
    Returns:
        torch.Tensor: the vector of the image in the out_layer
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    
    #Create a PyTorch Variable with the transformed image
    img = Variable(img_tensor).unsqueeze(0)
    
    # Create a vector of zeros that will hold the feature vector of the image. 
    # The 'avgpool' layer of the ResNet50 has an output size of 2048
    my_embedding = torch.zeros(2048)
    
    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    
    # Attach that function to our selected layer
    h = output_layer.register_forward_hook(copy_data)
    
    # Run the model on our transformed image
    net(img)
    
    # Detach our copy function from the layer
    h.remove()
    
    # Return the feature vector
    return my_embedding    

def CreateTestDataframe(index_path, label_path, broden_dir, percent_threshold):

    """
    Creates a matrix of images and classes for test data
    
    Args:
    index_path = path to the index.csv file in the Broden dataset
    label_path = path to the label.csv file in the Broden dataset
    broden_dir = path to the Broden dataset directory
    percent_threshold (int): the percentage of an image which a label must cover in the picture to be output as label
                            If the label covers less than the threshold it will not be regarded as label
    
    Returns: Dataframe
    
    """
    
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import math
    import pandas as pd
    from tqdm import tqdm_notebook
    import nbimporter
    
    from p1_GetClassesFromImage import GetClassesFromImage
    from p1_AddClassToMatrix import AddClassToMatrix
    
    
    image_dir = os.path.join(broden_dir, 'images/')

    index_df = pd.read_csv(index_path)
    label_df = pd.read_csv(label_path)
    
    #select only the images which are meant for testing
    test_df = index_df.loc[index_df.split == 'val',:]
    test_df.reset_index(inplace=True)
    
    class_list = label_df.loc[:,'name'].tolist() # This column stores all the labels in the images
    
    amount_of_classes = len(class_list) # amount of columns
    
    amount_of_pictures = len(test_df.loc[:, 'image']) # amount of rows
    
    # create a matrix to store the pictures with each concept
    classes_matrix = np.zeros(shape=(amount_of_pictures, amount_of_classes+1)) # add an additional column
    
    # This is the main for loop in which every image is parsed for classes 
    for image in tqdm_notebook(test_df.loc[:, 'image']): # Loop through every image in the Broden dataset

        row = test_df.index[test_df.image == image].values # get the row number of the image in the dataframe
        
        # read the columns containing the per-pixel label images and the scene and texture class
        color_img = test_df.loc[row, 'color'].values[0] 
        object_img = test_df.loc[row, 'object'].values[0]
        part_img = test_df.loc[row, 'part'].values[0]
        material_img = test_df.loc[row, 'material'].values[0]
        scene = test_df.loc[row, 'scene'].values[0]
        texture = test_df.loc[row, 'texture'].values[0]

        if isinstance(color_img, str): #check if the 'color' column contains a string, if yes, add the color classes to the matrix
            color_imgs = color_img.split(';')
            for c_img in color_imgs:
                colors = GetClassesFromImage(c_img, image_dir, percent_threshold)
                if np.any(colors):
                    classes_matrix = AddClassToMatrix(colors, row, classes_matrix)   

        if isinstance(object_img, str): #check image for object classes
            object_imgs = object_img.split(';')
            for o_img in object_imgs:
                objects = GetClassesFromImage(o_img, image_dir, percent_threshold)
                if np.any(objects):
                    classes_matrix = AddClassToMatrix(objects, row, classes_matrix)
                                                      
        if isinstance(part_img, str):                                     
            part_imgs = part_img.split(';')
            for p_img in part_imgs:
                parts = GetClassesFromImage(p_img, image_dir, percent_threshold)
                if np.any(parts):
                    classes_matrix = AddClassToMatrix(parts, row, classes_matrix)

        if isinstance(material_img, str): #check image for material classes
            material_imgs = material_img.split(';')
            for m_img in material_imgs:
                materials = GetClassesFromImage(m_img, image_dir, percent_threshold)
                if np.any(materials): #check if the the numpy array contains any values
                    classes_matrix = AddClassToMatrix(materials, row, classes_matrix)

        if not math.isnan(scene):
            # the scene will only contain one number, this can be directly linked to the matrix
            classes_matrix[row, int(scene)] = 1

        if isinstance(texture, str):
            texture_list = texture.split(';')
            for tex in texture_list:
                classes_matrix[row, int(tex)] = 1
    
    # convert the matrix to a dataframe and add labels to the columns
    test_data = pd.DataFrame(classes_matrix)
    labels = label_df.loc[:, 'name'].tolist()
    labels.insert(0, 'image')    
    test_data.columns = labels
    
    images = test_df.loc[:, 'image'].tolist()
    test_data.loc[:, 'image'] = images
    return test_data

def AddClassToMatrix(classes_in_img, row_in_matrix, class_m):
    
    """Adds the class numbers to the class matrix
    
    Args:
    classes_in_img: np.array containing the class numbers
    row_in_matrix: the row in the matrix represents the picture, this should be the same as the row of the image in index_df
    class_m: matrix in which the images are linked to the classes
    
    Returns: matrix
    """
    
    import numpy as np
    
    for num in np.nditer(classes_in_img):
        class_m[row_in_matrix, num] = 1
    
    return class_m

def CreateTrainingDataframe(index_path, label_path, broden_dir, percent_threshold):

    """
    Creates a matrix of images and classes for training data
    
    Args:
    index_path = path to the index.csv file in the Broden dataset
    label_path = path to the label.csv file in the Broden dataset
    broden_dir = path to the Broden dataset directory
    percent_threshold (int): the percentage of an image which a label must cover in the picture to be output as label
                            If the label covers less than the threshold it will not be regarded as label
    
    Returns: matrix
    
    """
    
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import math
    import pandas as pd
    from tqdm import tqdm_notebook
    import nbimporter
    
    from p1_GetClassesFromImage import GetClassesFromImage
    from p1_AddClassToMatrix import AddClassToMatrix
    
    
    image_dir = os.path.join(broden_dir, 'images/')

    index_df = pd.read_csv(index_path)
    label_df = pd.read_csv(label_path)
    
    #select only the images which are meant for training
    training_df = index_df.loc[index_df.split == 'train',:]
    
    class_list = label_df.loc[:, 'name'].tolist() # This column stores all the labels in the images

    amount_of_classes = len(class_list) # amount of columns
    amount_of_pictures = len(training_df.loc[:, 'image']) # amount of rows
    
    # create a matrix to store the pictures with each concept
    classes_matrix = np.zeros(shape=(amount_of_pictures, amount_of_classes+1)) # add an additional column
    
    
    # This is the main for loop in which every 
    for image in tqdm_notebook(training_df.loc[:, 'image']): # Loop through every image in the Broden dataset

        row = training_df.index[training_df.image == image].values # get the row number of the image in the dataframe
        
        # read the columns containing the per-pixel label images and the scene and texture class
        color_img = training_df.loc[row, 'color'].values[0]
        object_img = training_df.loc[row, 'object'].values[0]
        part_img = training_df.loc[row, 'part'].values[0]
        material_img = training_df.loc[row, 'material'].values[0]
        scene = training_df.loc[row, 'scene'].values[0]
        texture = training_df.loc[row, 'texture'].values[0]
        
        if isinstance(color_img, str): #check if the 'color' column contains a string, if yes, add the color classes to the matrix
            color_imgs = color_img.split(';')
            for c_img in color_imgs:
                colors = GetClassesFromImage(c_img, image_dir, percent_threshold)
                if np.any(colors):
                    classes_matrix = AddClassToMatrix(colors, row, classes_matrix)

        if isinstance(object_img, str): #check image for object classes
            object_imgs = object_img.split(';')
            for o_img in object_imgs:
                objects = GetClassesFromImage(o_img, image_dir, percent_threshold)
                if np.any(objects):
                    classes_matrix = AddClassToMatrix(objects, row, classes_matrix)
        
        if isinstance(part_img, str): #check image for part classes
            part_imgs = part_img.split(';')
            for p_img in part_imgs:
                parts = GetClassesFromImage(p_img, image_dir, percent_threshold)
                if np.any(parts):
                    classes_matrix = AddClassToMatrix(parts, row, classes_matrix)

        if isinstance(material_img, str): #check image for material classes
            material_imgs = material_img.split(';')
            for m_img in material_imgs:
                materials = GetClassesFromImage(m_img, image_dir, percent_threshold)
                if np.any(materials): #check if the the numpy array contains any values
                    classes_matrix = AddClassToMatrix(materials, row, classes_matrix)

        if not math.isnan(scene):
            classes_matrix[row, int(scene)] = 1
        
        if isinstance(texture, str):
            texture_list = texture.split(';')
            for tex in texture_list:
                classes_matrix[row, int(tex)] = 1
    
    # convert the matrix to a dataframe and add labels to the columns
    training_data = pd.DataFrame(classes_matrix)
    labels = label_df.loc[:, 'name'].tolist()
    labels.insert(0, 'image')    
    training_data.columns = labels
    
    images = training_df.loc[:, 'image'].tolist()
    training_data.loc[:, 'image'] = images
    return training_data

class SonDataset(Dataset):
    
    def __init__(self, updated_csv_path, img_path, transform = None):
        self.img_path = img_path
        self.votes_df = pd.read_csv(updated_csv_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        self.img_score = self.votes_df.loc[idx, 'Average']
        self.img_name = self.votes_df.loc[idx, 'ID'].astype(str)
        self.img_file = []
        
        for directory, _ , _ in os.walk(self.img_path):
            self.img_file.extend(glob.glob(os.path.join(directory, self.img_name + '.jpg')))
            
        im = Image.open(self.img_file[0])
        img_as_img = im.convert('RGB')
        
        if self.transform:
            img_as_img = self.transform(img_as_img)
            
        sample = {'image' : img_as_img,
                  'image_name' : self.img_name,
                  'image_path' : self.img_file[0],
                  'image_score' : self.img_score}
        return sample
    
    def __len__(self):
        return len(votes_df)
        
def MakeVectorDictionary(nn, out_layer, dataset, index_values, file_name=None):
    """
    Creates a dictionary with the row indices as key and the output tensor as value. The CNN is set to evaluation mode
    after which every image is run through the CNN. The output tensor of the 'avgpool' layer is stored as value in the 
    dictionary. After every image has been passed through the network, the dictionary is saved on the local harddrive.
    
    Args:
        nn (torchvision.model): a CNN which has a 'avgpool' layer before the fully connected layers
        out_layer: a layer of the nn, from which the output will be extracted
        dataset (torch.utils.data.Dataset): a dataset made with the function 'BrodenDataset'
        index_values (list): a list of integers refering to the row indices of the images labelled with a specific concept
                            in the training dataframe
        file_name (.pickle): default is None, the dictionary will not be stored.
                            If specified, the name with which the dictionary will be stored on the local harddrive. 
                            It must end with '.pickle'. The file will be stored in the data folder
        
    
    Returns:
        dictionary
    """
    
    #check if a GPU is available, otherwise run it on CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nn.to(device)
    # set the neural network to evaluation mode
    nn.eval()
    
    # create a dictionary in which the tensors will be stored
    vector_dict = {}
    
    # loop through every index, push the image belonging to the index through the network and extract the tensor from
    # the desired output layer
    for idx in tqdm_notebook(index_values):
        try:
            sample = dataset[idx]
            img = sample['image']
            name = sample['image_name']
            path = sample['image_path']
            if torch.cuda.is_available():
                img = img.float().cuda()
            else:
                img = img.float()
            vector_img = GetVector(img, nn, out_layer)
            vector_dict[str(idx)] = [name, path, vector_img]
        except:
            print('index:', idx, '\n', 'name:', name)
            
    # write the dictionary to the local harddrive
    if file_name is not None:
        with open(os.path.join('../data/', file_name), 'wb') as handle:
            pickle.dump(vector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return vector_dict    
    