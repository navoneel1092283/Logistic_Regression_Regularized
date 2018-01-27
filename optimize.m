function [theta,cost]=optimize(X,y,initial_theta,lambda)

	options = optimset('GradObj', 'on', 'MaxIter', 400);
	[theta, cost] =fminunc(@(t)(costFunctionReg(t, X, y,lambda)), initial_theta, options);
	predict(theta,X,y);
end
	


