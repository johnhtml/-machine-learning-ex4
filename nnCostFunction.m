function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); # Number of rows (features)
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------
% Part 1:
%   Theta1 has size 25 x 401
%   Theta2 has size 10 x 26

%   The cost of all the labels over the i sample
%       The vectors of the labels must be initialized

% how to use k labels
y_labels = eye(num_labels);

% FeedFordward
X = [ones(m, 1) X];

%   25*401x401*5000 = 25*5000
zValuesSecondLayer = Theta1*X';% ??
aValuesSecondLayer = sigmoid(zValuesSecondLayer);
aValuesSecondLayer = [ones(1, m); aValuesSecondLayer];

%   10*26x26*5000 = 10 * 5000
%   First column: is the map from the first sample
%   ith column: is the map from the first input sample
zValuesOutputLayer = Theta2*aValuesSecondLayer;
aValuesOutputLayer = sigmoid(zValuesOutputLayer);

% Iterating over the cost function
% 10*10x10*5000 = 10*5000
% 1*10x10*1 = 1
for i=1:m
    ym =  y(i,1);
    y_label = y_labels(ym,:);
    J = J - y_label*log(aValuesOutputLayer(:,i)) - (1-y_label)*log(1-aValuesOutputLayer(:,i));
endfor
J = J/m;

% Regularization of cost function
Theta1Temp = Theta1(:,2:end);
Theta2Temp = Theta2(:,2:end);
regularization = lambda/(2*m)*(sum(sum(Theta1Temp.^2)) + sum(sum(Theta2Temp.^2)));
J = J + regularization;

% -------------------------------------------------------------
% Part 2
% delta3
delta3 = zeros(m,num_labels);
for i=1:m
    ym =  y(i,1);
    y_label = y_labels(ym,:);
    delta3(i,:) =  aValuesOutputLayer(:,i)' - y_label;
endfor

% delta2
delta2 = (delta3*Theta2(:,2:end)).*sigmoidGradient(zValuesSecondLayer');

% Δ
Delta2 = zeros(size(Theta2));
Delta2 = Delta2 + delta3'*aValuesSecondLayer';

Delta1 = zeros(size(Theta1));
Delta1 = Delta1 + delta2'*X;

% Gradients
Theta1_grad = 1/m*Delta1;
Theta2_grad = 1/m*Delta2;

% -------------------------------------------------------------
% Part 2
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2Temp;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1Temp;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
