function[predict,ytest,mtest,theta]=traintest(X,y,n,data,lambda)
	ytest = data(:, n+1);
	[mtest,abc] = size(ytest);
	Xtest = data(:,1:n);
	Xtest = [ones(mtest,1) Xtest];
	initial_theta = zeros(n + 1, 1);
	[theta,cost]=optimize(X,y,initial_theta,lambda);
	predict=predict2(theta,Xtest);
end



