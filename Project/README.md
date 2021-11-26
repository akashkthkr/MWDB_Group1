1. get_color_moment_features(image)
    Introduction - This function takes an image of size 64*64 creates blocks of size 8*8 and then calls above three functions and finally returns a python dictionary having values Mean, Standard Deviation and Skewness.

2. get_gradient_feature(image)
    Introduction - This function is taking an image of size 64*64 and reshapes the image to 128*64 and then consumes image-scikit library, using its hog function for calculation HOG features. This function return a 1-D vector of size 3780.

3. get_local-binary_pattern(image)
    Introduction - This function is taking an image of size 64*64 and  consumes image-scikit library, using its local binary pattern image function for calculation HOG features.
    Here the return value is 2-D vector with each cell containing binary patten value. Hence histogram is consructed and thus the final return value of this function return a 1-D vector of size 26(number of points=24 +2).

4. get_l_norm_similarity(vector1, vector2,li)
    Introduction - This function computes the lnorm distance between two vectors vector1 and vector based on li value. Here li value is always going to be 1 as we are using it as manhattan distance

5. get_l_norm_similarity_from_merged(source, features, similarity-model, li, size, dataset)
    Introduction - This function takes the source image, features of all the images, similarity model on which this distance metric needs to be executed and the value of li, in this project it's always going to be 1 as we are always using it as manhattan distance.
    Here we are running a loop over all the images fetching respective feature model features and then we are feeding it to the get-l-norm-similarity function. This function returns top k similar images to the source image based on manhattan distance.

6. get_chi_square_similarity(vector1, vector2)
    Introduction - This method returns the chi-square distance between the two vectors having ELBP values.

7. get_chi_square_similarity_from_merged(source, features, size, dataset, similarity-model)
    Introduction - This function takes the source image, features of all the images. Here we are running a loop over all the images fetching ELBP features and then we are feeding it to the get-chi-square-similarity function. This function returns top k similar images to the source image based on chi-square distance.
    
8. get_weighted_similarity(color_moment, elbp, hog)
    Introduction - This method takes distance metric values of color moment , Extended Local binary pattern, and history of gradient and then takes 50 percent of Color moment, 20 percent of ELBP and 30 percent of HOG and combines the result and returns the value.

9. get-merged-similarity(source, features,size,dataset, similarity-model)
    Introduction - This function takes source image and features of all the images and runs a loop over all the images, extracting all three feature models and the respective similarity score is calculated. Then these similarity score is fed to get-weighted-similarity function and calculates the final score.
    This function returns top k similar images to the source image based on this solution

10. get-features(dataset)
    Introduction - This is a generic function for extracting all three features(Color Moment, ELBP, HOG) of all the images finally it is returning the result in the form of dictionaries. It is taking image data as input

11. load-dataset-from-folder(path, type='.png')
    Introduction - This method is taking dataset folder path creating image type as png and based on that extracting all the images from the dataset.

12. calculate-similarity(source, features, model, size)
    Introduction - This is a generic function for applying similarity measures on respective feature model. It is taking source image, features of all these images, feature model type and nearest similar count. Based on Model type respective similarity function is getting called. Finally the top k similar images to the source image are going to be returned.

13. write-result(source-path, result-list)
    Introduction - This function is taking source folder path and result folder path and based on that top similar images are getting pasted inside the result folder.

14. blockshaped(arr, nrows, ncols)
    Introduction - This function is taking image of type 64*64 and number of rows and columns as input. Based on these rows and columns image is getting converted to blocks and here block is of size 8*8 as we are always providing rows and column values as 8.

15. create-directory(path)
    Introduction -  This function is taking destination path where results are going to be stored as input and then creating a director at this location.

16. get-normalized-image(image)
    Introduction - This function is normalizing the cell values of an image from 0 to 1 as values can be from 0 to 255

17. merge-color-moment(features)
    Introduction -  This function is taking the color moment features of an image as input and then it concatenates the Mean, standard deviation and skewness of an image and then returns a list.

18. get-merged-feature-list(features))
    Introduction - This function is taking features of all the images and then loops over these images, fetches the color moment feature and then calls merge-color-moment for merging mean, standard deviation and skewness. Finally it returns a dictionaries of images with values corresponds to list of merged features.

19. store-data(path, source-image, model, similar-image-size)
    Introduction - All the features and similarities measures are called through this function. This function takes dataset path, source image , feature extraction model and k value(most similar images) as input and based on that extracts features and similarity score and stores tha data in json format.

20. get-normalized-vector(dict)
    Introduction - If the values of a vector dict is greater than 1 then this function normalizes the value from 0 to 1
