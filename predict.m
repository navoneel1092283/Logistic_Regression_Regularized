function predict(theta, X,y)


	m = size(X, 1); % Number of training examples


	p = zeros(m, 1);
	h = X*theta;
	hs= sigmoid(h);



	for i=1:m
		if hs(i)>=0.5,
			p(i)=1;
		elseif hs(i)<0.5,
			p(i)=0;
	end;
	end;
	
	disp(sprintf('Training Accuracy: %f',mean(double(p == y)) * 100));
end



