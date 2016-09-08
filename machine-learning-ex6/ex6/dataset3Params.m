function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

%initialize C and sigma
optimized_C_index = 1;
optimized_sigma_index = 1;
model = svmTrain(X, y, steps(optimized_C_index), @(x1, x2) gaussianKernel(x1, x2, steps(optimized_sigma_index)));
predictions = svmPredict(model, Xval);
optimized_error = mean(double(predictions ~= yval));

%start the loop to choose optimized C and sigma
for i = 1:8
    for j = 1:8
        if i == 1 && j == 1
            continue;
        end
        model = svmTrain(X, y, steps(i), @(x1, x2) gaussianKernel(x1, x2, steps(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error < optimized_error
            optimized_C_index = i;
            optimized_sigma_index = j;
            optimized_error = error;
        end;
    end;
end;

C = steps(optimized_C_index);
sigma = steps(optimized_sigma_index);


% =========================================================================

end
