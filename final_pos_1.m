function cf=final_pos_1(predict,ytest,mtest)
	tp=0;
	tn=0;
	fp=0;
	fn=0;
	for i=1:mtest
		if(predict(i)==1 && ytest(i)==1)
			tp=tp+1;
		elseif(predict(i)==1 && ytest(i)==0)
			fp=fp+1;
		elseif(predict(i)==0 && ytest(i)==1)
			fn=fn+1;
		else
			tn=tn+1;
		end;
	end;
	precision=tp/(tp+fp);
	recall=tp/(tp+fn);
	f1score=2*(recall*precision)/(recall+precision);
	accuracy=((tp+tn)/(tp+tn+fp+fn))*100;
	disp(sprintf('Accuracy=%f',accuracy));
	disp(sprintf('Precision=%f',precision));
	disp(sprintf('Recall=%f',recall));
	disp(sprintf('F1Score=%f',f1score));
	cf=[tp,fn;fp,tn];
end
			
			

			



