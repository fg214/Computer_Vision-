training_matrix=strings(0,0);
for i = 1:40 
    %open file 
    fileID = fopen('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\computer vision cw\trainclasses.txt');
    %textscan/fscanf to scan all the images in the training class
    A = textscan(fileID, '%s');
    %close file
    fclose(fileID);
    %add all of these to a matrix, instead of writing out file paths individually
 
    training_matrix = [training_matrix;A{1}{i}];
end
%set path to the training folders only, since there are overlapping
%animals of the test class in the training class
f = fullfile("\\smbhome.uscs.susx.ac.uk\fg214\Desktop\computer vision cw\Animals_with_Attributes2\JPEGImages", training_matrix);
 
 
    %imageDatastore
imds = imageDatastore(f, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test class
test_matrix = [];
test_matrix=strings(0,0);
for i_test = 1:10 
    %open file 
    fileID_test = fopen('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\computer vision cw\Animals_with_Attributes2\testclasses.txt');
    %textscan/fscanf to scan all the images in the training class
    A_test = textscan(fileID_test, '%s');
    %close file
    fclose(fileID_test);
    %add all of these to a matrix, instead of writing out file paths individually
    test_matrix = [test_matrix;A_test{1}{i_test}];
end
%set path to the training folders only, since there are overlapping
%animals of the test class in the training class
f_test = fullfile("\\smbhome.uscs.susx.ac.uk\fg214\Desktop\computer vision cw\Animals_with_Attributes2\JPEGImages", test_matrix);
 
 
%imageDatastore
imds_test = imageDatastore(f_test, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%bag of feautures 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%go through the binary predicate matrix 
b_predicate_matrix = dlmread('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\computer vision cw\Animals_with_Attributes2\predicate-matrix-binary.txt');

% if th

big_matrix = zeros(numpartitions(imds),85);
counter_start = 0;
counter_end = 0;
to_delete = [25; 39; 15; 6; 42; 14; 18; 48; 34; 24];
b_predicate_matrix(to_delete,:) = [];

for i = 1:40
   %a function which counts the # of images in each animal
  labelcounter = countEachLabel(imds);
  %number of images in A SPECIFIC ANIMAL folder
  x = labelcounter.Count(i);
   %starting counter. this is the value of the amount of pictures accumalated after every iteration 
   counter_start = counter_end + 1; 
   counter_end = counter_end + x;
   for y = counter_start:counter_end 
       %iterating the predicate matrix using the variable z
       v = b_predicate_matrix(i,:); 

       %copy the row of the predicate matrix y times into the big matrix
       big_matrix(y,:) = v;
   end
end

big_matrix;
fprintf('starting bag of features')
bag = bagOfFeatures(imds, 'StrongestFeatures', 0.8, 'VocabularySize', 500, 'Verbose', true, 'PointSelection', 'detector','Upright', true);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%encode bag first, so it can be passed as a vector 
featureVector = encode(bag, imds);

%pass the bag vector and big matrix to the fitcsvm function
%for every animal row in your big matrix, associate it with an attribute 
%for every attribute, 1-85, 
model = cell(85,1);
for a = 1:85
        model{a} = fitSVMPosterior(fitcsvm(featureVector, big_matrix(:,a)));  
end
%encode the test_features into a vector, 
test_feauture_vector = encode(bag, imds_test);
%pass this vector into the 'predict' function: 
%predict takes in two parameters, the svm model produced, and the test_feautures vector


%percentage of it being a 0 vs percantage of it being a 1
scores = [];
%LABEL:vector of binary value that represents the 2540 images by the 85 attributes shows the predicted value of each image having one of these attributes
imagexattribute = [];
imagexscores = [];
label = [];
%aa is indexing the image 
for aa = 1:(numpartitions(imds_test))
    %A indexing the 85 attributes
    for a = 1:85
        [label, score1] = predict(model{a}, test_feauture_vector(aa,:));
        assert(all(sum(score1, 2) ==1 ));
        %takes in the position of image(aa) and the atribute(a)
        imagexattribute(aa,a) = label;
        %takes in the position of the image(aa) and the score(probability of the attribute(a) being 1)
        imagexscores(aa,a) = score1(2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TEST PREDICATE MATRIX
%binary predicate matrix for test classes
b_predicate_matrix_test = dlmread('\\smbhome.uscs.susx.ac.uk\fg214\Desktop\computer vision cw\Animals_with_Attributes2\predicate-matrix-binary.txt');
big_matrix_test = zeros(numpartitions(imds_test),85);
counter_start_2 = 0;
counter_end_2 = 0;
to_delete = [25; 39; 15; 6; 42; 14; 18; 48; 34; 24];
delete_training = [1;2;3;4;5;7;8;9;10;11;12;13;16;17;19;20;21;22;23;26;27;28;29;30;31;32;33;35;36;37;38;40;41;43;44;45;46;47;49;50];
b_predicate_matrix_test(delete_training,:) = [];

for i = 1:10
   %a function which counts the # of images in each animal
  labelcounter_2= countEachLabel(imds_test);
  %number of images in A SPECIFIC ANIMAL folder
  x_2 = labelcounter_2.Count(i);
   %starting counter. this is the value of the amount of pictures accumalated after every iteration 
   counter_start_2 = counter_end_2 + 1; 
   counter_end_2 = counter_end_2 + x_2;
   for y_2 = counter_start_2:counter_end_2
       %iterating the predicate matrix using the variable z
       v_2 = b_predicate_matrix(i,:); 
       %copy the row of the predicate matrix y times into the big matrix
       big_matrix_test(y_2,:) = v_2;
   end
end
b_predicate_matrix_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%class probability
%create a matrix: each image(2540) x 85 attributes  
%%checks to see the 85 attributes of that image, 
%if that image, of a chimpanzee is: black, big, furry etc. or not
imagaxattribute;

counter_start_2 = 0;
counter_end_2 = 0;
bestclass = 0;
bestclassposition = 0;
bestclass_matrix = strings(numpartitions(imds_test), 1);


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %class probability
   position2 = [];
for aa = 1:(numpartitions(imds_test))
    %features of the single image
    p_matrix = [];
    %encode a single image from the test feauture vector
    feature_vector_test = encode(bag, readimage(imds_test, aa));
    %indexing the 10 test classes
    for yy = 1:10 
        %indexing each row from the predicate matrix of the test classes
        testrow2 = b_predicate_matrix_test(yy,:);
    %A indexing the 85 attributes
        for a = 1:85
            %using predict function on th feature vector 
            [label, score1] = predict(model{a},  feature_vector_test);
            %if the attribute of the image is a 0, access the score table with the probability of that attribute being a 0
            if testrow2(a) == 0
                p_matrix = [p_matrix;score1(1)];
            else
                %access the score matrix for the probability of 1
                 p_matrix = [p_matrix;score1(2)];
            end
        end
        p_matrix2 = [];
        p_matrix2(aa,yy) = prod(p_matrix);
    end
   [maxmax position] = max(p_matrix(aa,:));
   position2 = [position2; position];
   aa
end
 test_matrix(i)              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posmatrix=[6,8,2,4,3,1,9,10,7,5];
%counts each label in each of the 10 test images
testtable= countEachLabel(imds_test);
endCounter=0;
startCounter=1;
sumsum=[];
maxsum=[];
%for every image in the test class
for i=1: numpartitions(imds_test)
    endCounter= endCounter + testtable.Count(i);
    %for 10 classes in predicate
    for x=1:10
        %sum of probabilities from matrix
        sumsum= sum(position2(startCounter:endCounter)== posmatrix(x));
        %max of that sum
        maxsum=[maxsum; sumsum];
    end
end
accuracy=(sum(maxsum/numpartitions(imds_test)))*100;
%testcount1 = testtable(i, 'Count');
