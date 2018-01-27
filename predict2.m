function p=predict2(theta, X)


	m = size(X, 1); % Number of training examples


	h = X*theta;
	hs= sigmoid(h);


	for i=1:m
		if hs(i)>=0.5,
			p(i)=1;
		elseif hs(i)<0.5,
			p(i)=0;
	end;
end
	
