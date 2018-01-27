function g = sigmoid(z)

g = zeros(size(z));



den=exp(-z).+1;
g=1./den;





end