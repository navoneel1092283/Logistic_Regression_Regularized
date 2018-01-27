function p=evaluate(theta,X)
	X1=[ones(1,1) X];
	p=sigmoid(X1*theta);
end