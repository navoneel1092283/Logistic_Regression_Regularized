function [J, grad] = costFunctionReg(theta, X, y, lambda)
	m = length(y); % number of training examples
	J = 0;
	total=0;
	grad = zeros(size(theta));
	x=zeros(m,1);
	pl=zeros(m,1);
	cal=zeros(m,1);
	error=zeros(m,1);
	p=X*theta;
	for i=2:size(theta)
		total=total+theta(i)^2;;
	end;
	for i=1:m
		cal(i)=(-y(i)*log(sigmoid(p(i))))-((1-y(i))*log(1-sigmoid(p(i))))+((lambda/(2*m))*total);
	end;
	J=(1/m)*sum(cal);
	for i=1:m
		error(i)=sigmoid(p(i))-y(i);
	end;

	x=X(:,1);
	grad(1)=((1/m)*(sum(error.*x)));
	for j=2:length(theta)
		x=X(:,j);
		grad(j)=((1/m)*(sum(error.*x)))+((1/m)*(lambda*theta(j)));
		
	end;
end
