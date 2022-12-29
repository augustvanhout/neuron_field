import numpy as np
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import imageio
from keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore') 


def get_mnist_data(numbers_list):
    """
    Loads datasets of pictures of handwritten digits and labels. The outputs are 
    read-for-use in a NeuronField. Returns a tuple to be unpacked.
    
    Args:
        numbers_list: List of numbers. Will return mnist (records, labels) 
        of those numbers. For instance, passing [0, 1] will return a tuple of 
        shaped X data and y labels for mnist images of zeroes and ones.
        
    Try:
        X_train, y_train = get_mnist_data([0, 1, 2])
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 256
    y_train = pd.DataFrame(y_train)
    y_train = y_train[y_train[0].isin(numbers_list)]
    X_train = pd.DataFrame(X_train.reshape(-1, 784))
    X_train = X_train[X_train.index.isin(y_train.index)]

    X_train = np.array(X_train).reshape(-1, 28, 28)
    y_train = np.array(y_train)
    return X_train, y_train

def get_nearby_submatrix_3d(matrix, location_array, radius):
    """
    Returns a smaller subset of coordinates around a center point.
    Used in select_all_smlocs_3d().
    
    Args:
        matrix:
            Matrix. Try matrix = np.arange(1_000).reshape(10, 10, 10)
        location_array:
            3D Matrix of a single point in space. Try np.array([[4],[5],[6]])
        radius:
            Int. The distance around the point to return. A radius of 1 would return
            a 3x3x3 matrix.
    """
    x = location_array[0][0]
    y = location_array[1][0]
    z = location_array[2][0]
    x1 = int(max(x - radius, 0))
    x2 = int(min(x + radius, matrix.shape[0]))
    y1 = int(max(y - radius, 0))
    y2 = int(min(y + radius, matrix.shape[1]))
    z1 = int(max(z - radius, 0))
    z2 = int(min(z + radius, matrix.shape[2]))
    submatrix = matrix[max(x1 - 1, 0): x2,
                       max(y1 - 1, 0): y2, 
                       max(z1 - 1, 0): z2]
    return submatrix

def select_all_smlocs_3d(matrix): 
    """
    Takes a matrix and returns a list of all coordinates within it.
    
    Args:
        3D matrix. 
    """
    locs = []
    x = [i for i in range(matrix.shape[0])]
    y = [i for i in range(matrix.shape[1])]
    z = [i for i in range(matrix.shape[2])]
    for i in x:
        for j in y:
            for k in z:
                locs.append((x[i], y[j], z[k]))
    return locs

def select_all_locs_3d(matrix, center_array_coords, radius, verbose = False):
    """
    Takes a matrix and a point within it, and returns an itterable of all nearby coordinates within a radius.
    
    Args:
        matrix:
            3D matrix.
        center_array_coords:
            3D Matrix of a single point in space. Try np.array([[4],[5],[6]])
        radiius:
            Int. The distance around the point to return. A radius of 1 would return
            a 3x3x3 matrix.
        verbose:
            Boolean, default False. If true, walks through the steps to show inner workings of functions. 
    """
    submatrix = get_nearby_submatrix_3d(matrix, center_array_coords, radius)
    centralized_smlocs = select_all_smlocs_3d(submatrix)
    if verbose == True:
        print(type(center_array_coords))
        print("center_array_coords: \n",center_array_coords)
        print("submatrix: \n", submatrix)
        print("centralized_smlocs: \n",centralized_smlocs)
        print(matrix)
        print(matrix_loc)
    return [(max(tup[0] + center_array_coords[0] - radius, np.array([0])), 
                   max(tup[1] + center_array_coords[1] - radius, np.array([0])),
                   max(tup[2] + center_array_coords[2] - radius, np.array([0]))) for tup in centralized_smlocs]

def upflow_axon_extension(Neuron, NeuronField):
    """
    Extends a neuron's axon in an upwards direction relative to its cell body.
    This generally creates a flow of neuron firing.
    
    If there are no suitable location, the neuron moves the cell body and retries.
    
    Args:
        Neuron: neuron object. It's axon will be extended.
        NeuronField: neuronfield object. Required for checking axon locations,
            repositioning.
    """
    while Neuron.axon_loc is None:
        outer_locs = select_all_locs_3d(matrix = Neuron.matrix, 
                                            center_array_coords = Neuron.loc, 
                                            radius = Neuron.axon_length, 
                                            verbose = False)
        possible_locs = [loc for loc in outer_locs if 
                         np.linalg.norm(loc - np.array([loc for loc in Neuron.loc]))
                        > Neuron.axon_length - 1]
        if Neuron.neuron_type == "field":
            possible_locs = [loc for loc in possible_locs 
                             if loc[2] >= Neuron.loc[2] + Neuron.dend_radius
                             if loc[2] >= NeuronField.input_z]
        try:
            Neuron.axon_loc = possible_locs[np.random.choice(len(possible_locs))]
        except:
            Neuron.loc = np.array([[np.random.choice(NeuronField.field.shape[0], 1)],
                        [np.random.choice(NeuronField.field.shape[1], 1)], 
                        [np.random.choice(NeuronField.field.shape[2], 1)]])
            
def apply_activation_function(x, function):
    """
    Activation function is applied to each element of an itterable.
    Best results are with Relu.
    
    If "sigmoid" is selected, unstimulated neurons will fire since 
    sigmoid of 0 returns 0.5!
    
    Input: 
        X: itterable of numerics
        function: String
            Available Functions: elu, relu, sigmoid, tanh
    Output: Input values which have had the function applied to them
    """
    if function == "elu":
        x = np.exp(x)-1
    if function == "relu":
        x = x * (x > 0)
    if function == "sigmoid":
        x = 1/(1 + np.exp(-x))
    if function == "tanh":
        x = np.tanh(x)
    # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    #https://www.geeksforgeeks.org/implement-sigmoid-function-using-numpy/
    # https://www.geeksforgeeks.org/numpy-tanh-python/#:~:text=The%20numpy.,sinh(x)%20%2F%20np.
    return x

def custom_cell_model(Neuron, verbose = False):
    """ 
    An example of the structure required for customizing a cell's internal logic.
    
    Inputs are a list of values from nearby axons.
    axon_output should be numeric, preferably scaled between -1 and 1
    """
    inputs = Neuron.dendrite_input
    axon_output = None #try np.mean(inputs)
    
    if verbose == True:
        print(inputs)
        print(axon_output)
        
    return axon_output

class Neuron:
    def __init__(self, input_matrix, 
                 input_matrix_loc, 
                 matrix, 
                 loc, 
                 dend_radius, 
                 min_dends,
                 axon_length, 
                 neuron_type, 
                 neuron_id,
                 storage = None,
                 axon_extension_function = None,
                 cell_model_type = None,
                 activation_functions_list = None
                ):
        """
        Definition:
            Neurons are instantiated, placed in space, and made to interact with their neighbors. 
            
            They read their neighbors' values and then set their own value as a result. After instantiating a NeuronField, 
            functions are be called within that NeuronField to fill it with neurons.
            
            There are Input, Field, and Output neurons by default.     
                Input neurons are made to stimulate an area of the field without reading neighbors. By default, they take 
                an image and place it near the bottom of the field. 
                
                Field neurons exist in the 3D field and connect to a group of neighbor. Then, by default, they read their 
                neighbor's values and use a DNN to determine their own value.
                
                Output neurons are few (likely <5), are located near the top of the field, and take the mean of nearby activity.
                
            Neurons have two locations associated with them: Cell body location (loc) and axon location (axon_loc). The cell body 
            is the area around which neighbors will be read. The axon location is the point where the neuron will put its own 
            value. The axon is extended away from the center. Neurons cannot read their own axons.
            
                cell body ->  (  O--)-----o    <- axon 
                       radius ^     ^

        Args:
            input_matrix:
                Numpy array: The matrix object where the neuron will collect inputs from nearby axons.
            input_matrix_loc:
                Numpy array: The location of the neuron in the input matrix. (ie. np.array([[3], [2]]) in a 2d input image matrix.)
            matrix:
                Numpy array: The matrix object from which the neuron will extend its axon. For most neurons, it is the same as
                the input matrix.
            loc:
                Numpy array: The location of the neuron in the matrix. (ie. np.array([[3], [2], [7]]) in a 3d space.) 
            dend_radius:
                Numeric: The radius around the neuron where neighbors' axons will be identified.
            min_dends:
                Int: The number of connections to nearby axons. If too few are found during a search for neighbors, the dend_radius 
                will increase by 1 and the search will be repeated.
            axon_length:
                Numeric: The distance away from the cell the axon will be placed during axon extention. (i.e. a neuron in location
                np.array([[3], [3], [1]]) with an axon length of 4 might place its axon in ([[5], [4], [0]]) because it has a euclidean 
                distance of approximately 4 from the center of the neuron.)
            neuron_type:
                String: Default values are "Input", "Field", "Output"
            neuron_id:
                Int: Integer identifier for each neuron. Unique to all neurons, regardless of types.
            storage:
                Object: Any object (files, models, functions, anything) which the neuron references as neuron.strorage. Currently 
                not utilized.
            axon_extension_function:
                Object (function): 
                Defaults to upflow_axon_extension, which extends axons in the same general direction.          
        """

        self.input_matrix = input_matrix # the matrix in which the neuron will search for neighbors
        self.input_matrix_loc = input_matrix_loc # the location within this input matrix
        self.matrix = matrix # the matrix in which the cell body and axon are located
        self.loc = loc # the location within this matrix
        
        # the term "dendrite" is used when forming connections to nearby axons. 
        # dendrites are not objects. This term is just used for the sake of
        # mimicking the brain. Dendrites in this context are just a connection 
        # between a cell body's model's input node and another neuron's axon
        self.dend_radius = dend_radius # initial search radius for nearby axons
        self.min_dends = min_dends # if the neuron does not find this many neighbors, it will extend its search radius
        self.dend_list = [] # the list of IDs of neurons whose axons are read by this neuron
        self.dend_weights = [] # the list of corresponding weights to apply to the 
        
        # Axons are points in space extended away from neurons which carry that neuron's output.
        self.axon_length = axon_length # the axon will be placed this far from the cell using an axon_extension_function
        self.axon_loc = None # numpy array containing coordinates. When the cell 
        self.dendrite_input = np.nan # 
        self.nn_matrices = []
        self.axon_output = 0
        self.new_axon_output = 0
        
                
        self.neuron_type = neuron_type
        self.neuron_id = neuron_id
        
        self.cell_model_type = cell_model_type
        if self.neuron_type not in ["input", "output"] and self.cell_model_type is None:
            self.cell_model_type = "DNN"
            self.weights_biases_list = []
            self.cell_model_layers = 1
            self.activation_functions_list = ["relu", "relu", "relu"]

        self.axon_extension_function = axon_extension_function
        if self.axon_extension_function is None:
            self.axon_extension_function = upflow_axon_extension
        # if no custom extension function passed, default to upflow
        
        self.storage = storage
    
    def extend_axon(self, NeuronField): 
        """
        Applies this neuron's axon_extension_function to itself and the specified NeuronField.
        Axon extension works by default. If a custom function is made but fails to extend the axon, 
        this function will throw an error.
        """
        while self.axon_loc is None: # while loop permits recursive attempts to relocate and extend axons
            self.axon_extension_function(self, NeuronField)
            assert self.axon_loc is not None, f"Neuron {self.neuron_id} in location {self.loc} failed to extend axon." 
                # if the function did not work, raise an error

    def find_nearby_axons(self, NeuronField, neighbor_types):
        """
        Sets this neuron's dend_list to a list of neuron_id numbers whose axons are within dendrite radius and 
        whose type is found within the passed list of neighbor types.
        If the number of neurons found does not meet this neuron's min_dends, the radius is increased by 1.
        
        Args:
            NeuronField: neuronfield object
            neighbor_types: list of strings. ie. ["input", "field"]
        """
        while len(self.dend_list) < self.min_dends:
            df_copy = NeuronField.neuron_df.copy()
            df_copy = df_copy[df_copy["id"] != self.neuron_id]
            if self.neuron_type == "output":
                df_copy = df_copy[df_copy["type"] != "input"]
            df_copy["dist"] = df_copy["axon_loc"].apply(lambda x :
                np.linalg.norm(np.array([loc for loc in x]) - np.array([loc for loc in self.loc]))
            )
            nearby_axons = list(set(df_copy[df_copy["dist"] < self.dend_radius]["id"].tolist()))
            if len(nearby_axons) < self.min_dends:
                self.dend_radius += 1
            else:
                self.dend_list = nearby_axons
                
    def read_dend_list(self, NeuronField):
        """
        Sets this neuron's dendrite_input to a list of values passed by the attached neighboring dendrites. Accesses 
        this information in the passed NeuronField.
        """
        self.dendrite_input = NeuronField.neuron_df[NeuronField.neuron_df["id"].isin(self.dend_list)]["axon_output"].tolist()
    
    def instantiate_cell_body(self):
        """
        Gives a chance to run code to set up any complicated internal workings of a neuron. By default, it sets up a basic numpy dense neural network
        with a single layer. This reads the dendrites and sets the axon value to the output of the neural network.
        
        If the neuron has had a custom cell_model_type passed to it during instantiation, modify the elif statement below to run whatever functions 
        needed.
        """
        if self.cell_model_type == "DNN":
            self.weights_biases_list = [[np.random.randint(20, 80, (len(self.dend_list), len(self.dend_list))) / 100, np.zeros(len(self.dend_list),)]
                                        for layer in np.arange(self.cell_model_layers)]
            self.weights_biases_list.append(
                [np.random.randint(40, 100, (len(self.dend_list), len(self.dend_list))) / 100, np.zeros(1,)]
                    )
        elif self.cell_model_type:
            pass # if steps are required to set up each neuron's custom model, define them here.
            
    def adjust_weights(self, s):
        """ 
        Utility for neurons with "DNN" models.
        Randomly adjust the weights in each layer of a neuron
        Inputs: 
            s: Float. All weights will become the sum of themselves and a number between -s and s.
        """
        z_adjustment = self.loc[2] / (self.matrix.shape[2])
        weight_adjustment = [[np.random.randint(-s * 100, s * 100, (len(self.dend_list), len(self.dend_list))) / 100, np.zeros(len(self.dend_list),)] 
                             for layer in np.arange(self.cell_model_layers)]
        weight_adjustment.append(
            [np.random.randint(-s * 100, s * 100, (len(self.dend_list), len(self.dend_list))) / 100, np.zeros(1,)]
                )
        for layer in range(len(self.weights_biases_list)):
            self.weights_biases_list[layer][0] = self.weights_biases_list[layer][0] + weight_adjustment[layer][0] 
    
    def firing_cycle(self):
        """
        By default, performs a feed-forward operation on the DNN, setting the axon value with the output.
        
        If a custom cell_model_type has been passed during this neurons instantiation, modify the elif statement below to make this neuron 
        function as intended during firing.
        """
        if self.cell_model_type == "DNN":
            assert len(self.activation_functions_list) == len(self.weights_biases_list) + 1, \
                    f"The number of activation functions ({len(self_activation_functions)}) \
                    was shorter than the number of weight/bias sets for this neuron's model ({len(self.weights_biases_list) + 1}). \
                    Please pass a list matching the number of weight sets, ie ['relu', 'relu', 'sigmoid']."
            self.nn_matrices = []
            self.nn_matrices.append(self.dendrite_input)
            i = 0
            for weight, bias in self.weights_biases_list:
                temp = np.dot(self.nn_matrices[i], weight)+bias
                self.nn_matrices.append(apply_activation_function(temp, self.activation_functions_list[i]))
                i += 1
            self.new_axon_output = apply_activation_function(temp[0], self.activation_functions_list[i])
            # known issue. Final activated output is a list of identical values. First is referenced here as output.
        elif self.cell_model_type:
            self.new_axon_output = custom_cell_code(self, verbose = False)

    def set_new_axon_value(self):
        """
        Sets this neuron's axon value to the next value (saved during firing). This is done to make sure all neurons read their neighbors,
        fire, and then change their values in unison.
        """
        self.axon_output = self.new_axon_output
        
        
class NeuronField:
    """
    Definition:
    
    NeuronFields hold neuron information and facilitate interactions. They are used to create a 3D matrix, fill it with Field neurons 
    which interact with one another, place input neurons for stimulating the bottom of the field, and placing output neurons at the top 
    for recording outputs.
    
    They also gather and hold information about the neurons and field, prediction and accuracy scores, and more.
    
    To get started, make a test folder in your directory and try the following:
    ```
    X_train, y_train = get_mnist_data([0,1])
    nf_test = NeuronField(input_size = 28, 
                      field_size = 28, 
                      output_size = 2,
                      k_neurons = 300, 
                      min_dendrites = 4, 
                      dendrite_radius = 4, 
                      axon_length = 7
                     )
    nf_test.initiate_field()
    nf_test.prediction_gif(X_train[0], steps = 20, output_path = "my_test_folder", prefix = "test_gif")
    # look for the _movie.gif file in the folder!
    ```
    
    Args:
        input_size:
            Int. The length of one side of a square image input matrix.
            (ie. for a 10x10 image matrix, input size would be 10.)
        field_size:
            Int. The length of one side of a three dimensuonal field matrix. Be default, this field
            will be twice this size in the dimension spanning input to output.
            (ie. field size of 28 would produce a 28x28x56 matrix. For this package's capabilities, this
            would be a relatively large matrix.)
        output_size: 
            Int. The number of output neurons. These will be distribured in an equidistant circle around the 
            center of the top of the field. Recommended to keep to numbers between 1-3, unless neurons are 
            extremely dense (> 3,000 neurons for 3+ outputs is safer.)
        k_neurons: 
            Int. The number of field neurons to instantiate in the model. These will be randomly places in 
            the field.
        min_dendrites:  
            Int. The minimum number of connections each neuron should have to neighbors.
        dendrite_radius: 
            Numeric. The euclidean distance within which each neuron will read neighbors. If a neuron does not 
            find enough neighbors, it will increase this radius for itself. If a neuron finds many neighbors, 
            it will have many dendrite connections.
        axon_length:
            Int. The approximate distance from the cell body each field neuron will extend its axon.
    """
    def __init__(self, 
                 input_size, 
                 field_size,
                 output_size, 
                 k_neurons, 
                 min_dendrites, 
                 dendrite_radius, 
                 axon_length):
        self.input_size = input_size
        self.field_size = field_size
        if output_size >= 5: 
            warnings.warn("In current implementation, output neurons are placed in the top of the field, and are distributed in \
            a ring. With more than 5 outputs, consider higher than average field neuron density since multiple output neurons may \
            begin reading from the same field neuron.")
        self.output_size = output_size
        self.k_neurons = k_neurons
        self.min_dendrites = min_dendrites
        self.dendrite_radius = dendrite_radius
        self.axon_length = axon_length
        
        self.image_id = 0
        self.step_count = 0
        self.neuron_id = 0
        
        self.input = np.array([None for i in range(0, input_size**2)], dtype = np.float64).reshape(input_size, input_size)
        self.field = np.array([None for i in range(0, (field_size**3) * 2)], dtype = np.float64).reshape(field_size, field_size, field_size * 2)
        self.output = np.array([None for i in range(0, output_size)], dtype = np.float64)
        
        self.input_z = self.field_size / 3
        
        self.neuron_dict = {
            "input" : {},
            "field" : {},
            "output" : {}
        }
        
        self.neuron_df = pd.DataFrame({
            "id" : [],
            "type" : [],
            "loc" : [],
            "axon_loc" : [],
            "axon_output" : [],
            "dendrite_input" : [],
            "dend_list" : [],
            "dend_weights" : []
        })
        
        self.output_df_list = []
        self.output_df = pd.DataFrame()
        self.accuracy_score = None
            
    def gather_neuron_data(self, neuron_types):
        """
        Generates a neuron_df of all neurons specified within the list of types.
        This is done automatically during the instantiation and before each prediction step.
        
        Try:
            this_field.gather_neuron_data["input", "field", "output"]
            this_field.neuron_df
            
        Args:
            neuron_types: List of strings. Neuron types to colelct information on.
        """
        self.neuron_df = pd.DataFrame({
            "id" : [],
            "type" : [],
            "loc" : [],
            "axon_loc" : [],
            "axon_output" : [],
            "dendrite_input" : [],
            "dend_list" : [],
            "dend_weights" : []
        })  
        for neuron_type in neuron_types:
            neuron_id_list = list(self.neuron_dict[neuron_type].keys())
            df = pd.DataFrame({
                "id" : [self.neuron_dict[neuron_type][n_id].neuron_id for n_id in neuron_id_list],
                "type" : [self.neuron_dict[neuron_type][n_id].neuron_type for n_id in neuron_id_list],
                "loc" : [self.neuron_dict[neuron_type][n_id].loc for n_id in neuron_id_list],
                "axon_loc" : [self.neuron_dict[neuron_type][n_id].axon_loc for n_id in neuron_id_list],
                "axon_output" : [self.neuron_dict[neuron_type][n_id].axon_output for n_id in neuron_id_list],
                "dendrite_input" : [self.neuron_dict[neuron_type][n_id].dendrite_input for n_id in neuron_id_list],
                "dend_list" : [self.neuron_dict[neuron_type][n_id].dend_list for n_id in neuron_id_list],
                "dend_weights" : [self.neuron_dict[neuron_type][n_id].dend_weights for n_id in neuron_id_list]
            })
            self.neuron_df = self.neuron_df.append(df)
            
    def reset_neuron_data(self, neuron_types):
        self.neuron_df = self.neuron_df[~self.neuron_df["type"].isin(neuron_types)]  
        
    def instantiate_input_neurons(self):
        """
        Instantiates input neurons on the bottom of the field, scaled in location to the size relationship between
        the input/field matrices. For the mnist dataset, with image sizes of 28x28, creates 784 neurons which 
        correspond to pixels in input image matrix.
        
        These neurons are not located on the field, but their axons are.
        """
        input_to_field_scale = self.field_size / self.input_size
        ycoord = 0
        for y in self.input:
            xcoord = 0
            for x in y:
                xc = np.floor(input_to_field_scale * xcoord)
                yc = np.floor(input_to_field_scale * ycoord)
                self.neuron_dict["input"][self.neuron_id] = Neuron(
                     input_matrix = self.input, 
                     input_matrix_loc = (xcoord, ycoord), 
                     matrix = self.field, 
                     loc = np.array([[xc], [yc]]),
                     dend_radius = 0, 
                     min_dends = 1,
                     axon_length = 0, 
                     neuron_type = "input",
                     neuron_id = self.neuron_id)
                self.neuron_dict["input"][self.neuron_id].axon_loc = np.array([[xc], [yc], [self.input_z]])
                self.neuron_id += 1
                xcoord += 1
            ycoord += 1

    def instantiate_field_neurons(self, verbose = False):
        """
        Instantiates the number of field neurons specified for this NeuronField.
        Places these neurons randomly around the field, extends their axons, gathers their data,
        establishes neighbor connections, and prepares their cell bodies for firing.
        """
        print(f"Found {len(self.neuron_dict['input'].keys())} input neurons.")
        for k in range(self.k_neurons):
            neuron_loc = np.array([[np.random.choice(self.field_size, 1)],
                   [np.random.choice(self.field_size, 1)], 
                   [np.random.choice(self.field_size * 2, 1)]])
            self.neuron_dict["field"][self.neuron_id] = Neuron(
                 input_matrix = self.field, 
                 input_matrix_loc = neuron_loc,
                 matrix = self.field,
                 loc = neuron_loc,
                 dend_radius = self.dendrite_radius,
                 min_dends = self.min_dendrites,
                 axon_length = self.axon_length, 
                 neuron_type = "field",
                 neuron_id = self.neuron_id)
            self.neuron_id += 1
        print(f"{self.k_neurons} field neurons instantiated.")
        for neuron_id in self.neuron_dict["field"].keys():
            self.neuron_dict["field"][neuron_id].extend_axon(self)
        print("Axons extended.")
        self.gather_neuron_data(["input", "field", "output"])
        neuron_count = self.neuron_df["id"].nunique()
        print(f"Data gathered from {neuron_count} neurons.")
        for neuron_id in self.neuron_dict["field"].keys():
            self.neuron_dict["field"][neuron_id].find_nearby_axons(NeuronField = self, neighbor_types = ["input", "field"])
            if verbose == True:
                if neuron_id % 100 == 0: 
                    print(neuron_id)
        print("Axon neighbors identified.")
        for neuron_id in self.neuron_dict["field"].keys():
            self.neuron_dict["field"][neuron_id].instantiate_cell_body()
        print("Cell body models instantiated.")

    def instantiate_output_neurons(self):
        """
        Instantiates the number of output neurons specified for this NeuronField. 
        Places these a few levels below the top of the field, equidistant from one another in a circle 
        around the center.
        """
        output_to_field_scale = self.field_size / self.output_size
        z_loc = (self.field_size * 2) - 2 # place the outpout neurons 2 levels from the top
        y_coord = 0
        center = np.array([[self.field_size / 2], [self.field_size / 2]])
        radius = self.field_size / 3
        angle = 360 / self.output_size
        angles = np.array([angle * point for point in range(1, self.output_size + 1)])
        coords = [np.array([[x * radius], [y * radius]]) + center for (x, y) in zip(np.cos(angles), np.sin(angles))]
        coords = [np.array([[(max(x[0], 0))], [max(y[0], 0)]]) for (x,y) in coords]
            # thanks to John Charles for this one. This is his math and my code. Jonathan.charles9494@gmail.com
        for i in range(self.output_size):
            neuron_loc = np.array([coords[i][0],
                          coords[i][1],
                          z_loc])
            self.neuron_dict["output"][self.neuron_id] = Neuron(
                 input_matrix = self.input, 
                 input_matrix_loc = neuron_loc, 
                 matrix = self.field, 
                 loc = neuron_loc,
                 dend_radius = 10,
                 min_dends = 1,
                 axon_length = 0, 
                 neuron_type = "output",
                 neuron_id = self.neuron_id)
            self.neuron_dict["output"][self.neuron_id].find_nearby_axons(self, ["field"])
            self.neuron_id += 1
        self.gather_neuron_data(["input", "field", "output"])
    
    def initiate_field(self):
        """
        Instantiates input, field, and output neurons. After this, the field is ready for prediction cycles.
        """
        self.instantiate_input_neurons()
        self.instantiate_field_neurons()
        self.instantiate_output_neurons()
    
    def place_image(self, image):
        """
        Sets the input neuron's axon values according to the image passed.
        
        In effect, creates an image on the bottom of the field matrix. The pixel values of the image become axon values 
        which nearby neurons read. Be sure to scale pixel values from 0-1.
        
        Args:
            image: 2D Matrix of axon values.
            (i.e. mnist's X_train data:  np.array(X_train[0]).reshape(-1, 28, 28))
        """
        self.image_id += 1
        counter = 0
        y_coord = 0
        for y in image:
            x_coord = 0
            for x in y:
                self.neuron_dict["input"][counter].axon_output = image[x_coord,y_coord]
                counter += 1
                x_coord += 1
            y_coord += 1
            
    def step(self):
        """
        One cycle of reading and firing for all neurons, plus output recording.
        
            First, gathers all neuron data. 
            Second, for all neurons, reads dendrite inputs and fires.
            Third, sets all new axon values.
            Fourth, records output neuron values.
            Fifth, Itterates this NeuronField's step count by 1.
        """
        self.gather_neuron_data(["input", "field", "output"])
        for neuron in self.neuron_dict["field"].keys():
            self.neuron_dict["field"][neuron].read_dend_list(self)
            self.neuron_dict["field"][neuron].firing_cycle()
        for neuron in self.neuron_dict["field"].keys():
            self.neuron_dict["field"][neuron].set_new_axon_value()
        for neuron in self.neuron_dict["output"].keys():
            self.neuron_dict["output"][neuron].read_dend_list(self)
        self.step_count += 1
        
    def record_output(self):
        """
        Creates a dataframe of basic output information. Done automatically during prediction
        cycles. After predicitons, try:
        
            this_field.output_df
            
        """
        df = pd.DataFrame({"image_id" : [self.image_id],
                           "step" : [self.step_count]})
        for n in self.neuron_dict["output"].keys():
            self.neuron_dict["output"][n].read_dend_list(self)
            self.neuron_dict["output"][n].axon_output = np.mean(self.neuron_dict["output"][n].dendrite_input)
            df[n] = [self.neuron_dict["output"][n].axon_output]   
        self.output_df_list.append(df)

    def reset_neurons(self, neuron_type = None):
        """
        Sets all axon outputs to 0. Done during firing, but useful in experimentation.
        If list of types passed, resets only those types. (i.e. this_field.reset_neurons(["output"]) )
        """
        if neuron_type:
            for neuron in self.neuron_dict[neuron_type].keys():
                self.neuron_dict[neuron_type][neuron].axon_output = 0
        else:
            for neuron_type in self.neuron_dict.keys():
                for neuron in self.neuron_dict[neuron_type].keys():
                    self.neuron_dict[neuron_type][neuron].axon_output = 0

    def prediction_sequence(self, image, steps, verbose = False):
        """
        Takes an image matrix, places it at the bottom as the values of the input neuron's axons, then
        performs a set number of firing cycles, and records output as a record in the output_df.
        
        Args:
            image:
                Matrix of size input_size x input_size. For an mnist image, this would be 28x28, with 
                values in the matrix being pixel darknesses.
            steps:
                Int. Number of firing cycles before the final output values are recorded.
            verbose:
                If set to true, prints the step number.
        """
        self.reset_neurons()
        self.step_count = 0
        self.place_image(image)
        for step in range(steps):
            self.step()
            if verbose == True:
                print("Step: ", step + 1)
        self.record_output()
        self.output_df = pd.concat(self.output_df_list)
        
    def generate_accuracy(self, y_data):
        """
        Adds accuracy information to the output_df using a vector of numeric labels. If the NeuronField just predicted the 
        first 10 records in a dataset, uses the first 10 values in the y_data label vector to calculate accuracy for each row.
        Used by predict function.
        
        Args:
            y_data: 1D matrix of numeric labels.
        """
        self.output_df["prediction"] = self.output_df[[i for i in self.neuron_dict["output"].keys()]].reset_index(drop = True).idxmax(axis = "columns")
        self.output_df["y"] = self.output_df["image_id"].apply(lambda x : y_data[x][0])
        self.output_df["y_neuron"] = self.output_df["y"].apply(lambda x : list(self.neuron_dict["output"].keys())[x])
        self.output_df["is_correct"] = self.output_df["prediction"] == self.output_df["y_neuron"]
        self.accuracy_score = self.output_df["is_correct"].mean()
        
    def predict(self, X_data, y_data, records, steps):
        """
        Passed an itterable of image matrices (X) and a vector of labels (y), performs firing cycles (steps) on a speficied number (records) of 
        those records. Automatically records outputs and assigns this NeuronField an accuracy_score.
        
        Args:
            X_data:
                Itterable of input x input sized matrices.
            y_data:
                Vector of numeric labels. These will be used in accuracy calculation.
            records:
                Number of records to perform prediction, starting from the top of the itterables.
            steps:
                Number of prediction firing cycles per record.
        """
        self.output_df_list = []
        for k in range(records):
            self.prediction_sequence(X_train[k], steps = steps, verbose = False)
        self.generate_accuracy(y_data)
        return self.output_df, self.accuracy_score
    
    def prediction_gif(self, image, steps, output_path, prefix = None):
        """
        Generates a gif of firing steps, with neurons activating in a wave across the field. Saves the 
        gif (and a *lot* of images) in a specified folder path. Does not record outputs.
        
        Args:
            image:
                Matrix of size input_size x input_size. For an mnist image, this would be 28x28, with 
                values in the matrix being pixel darknesses.
            steps:
                Int. Number of firing cycles before the final output values are recorded.
            output_path:
                String: path to folder where the function will dump images and assemble a gif.
                    (i.e. "my_gif_folder")
            prefix:
                String: prefix of files within the folder (followed by the step number and .png or .gif)
                    (i.e. "my_gif")
        """
        self.reset_neurons()
        self.step_count = 0
        self.place_image(image)
        gif_images = []
        for x in range(1, steps):
            self.gather_neuron_data(["input", "field", "output"])
            ax = plt.axes(projection='3d')
            df = self.neuron_df
            df1 = df[df["type"].isin(["field"])]
            df1["x"] = [int(loc[0]) for loc in list(df1["loc"])]
            df1["y"] = [int(loc[1]) for loc in list(df1["loc"])]
            df1["z"] = [int(loc[2]) for loc in list(df1["loc"])]
            ax.scatter3D(df1["x"], df1["y"], df1["z"], s = 20, c = df1["axon_output"], cmap='PuRd');
            
            df2 = df[df["type"].isin(["input"])]
            df2["x"] = [int(loc[0]) for loc in list(df2["axon_loc"])]
            df2["y"] = [int(loc[1]) for loc in list(df2["axon_loc"])]
            df2["z"] = [int(loc[2]) for loc in list(df2["axon_loc"])]
            ax.scatter3D(df2["x"], df2["y"], df2["z"], c = df2["axon_output"], cmap='Blues');
            
            df3 = df[df["type"].isin(["output"])]
            df3["x"] = [int(loc[0]) for loc in list(df3["loc"])]
            df3["y"] = [int(loc[1]) for loc in list(df3["loc"])]
            df3["z"] = [int(loc[2]) for loc in list(df3["loc"])]
            df3["sum"] = df3["dendrite_input"].apply(lambda x: np.mean(x))
            ax.scatter3D(df3["x"], df3["y"], df3["z"], c = df3["sum"], s = 200, cmap = "GnBu");
            
            filepath = f"{output_path}/{prefix}_{x}.png"
            plt.savefig(filepath)
            for n in range(4):
                gif_images.append(imageio.imread(filepath))
            self.step()
        imageio.mimsave(f"{output_path}/{prefix}_movie.gif", gif_images)
