/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package auxiliary;

import java.awt.Label;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 *
 * @author zhouguobing
 */
public class AdaBoost extends Classifier {

	int k; //turns
	ADecisionTree[] boost;
	double[] dataWeights;
	double[] classifierWeights;
	double[] errors;
	
    public AdaBoost() {
    	k=10;
    	boost = new ADecisionTree[k];
    	errors = new double[k];
    	classifierWeights = new double[k];
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {

    	int insNum = features.length;
    	int length = insNum/k;
    	dataWeights = new double[insNum];
    	
    	for(int i=0;i<insNum;i ++)
    		dataWeights[i] = 1.0/insNum;
    	
    	for(int turns=0;turns<k;turns ++) {
    		
	    	List<Object> tmpdata = new ArrayList<Object>();
	    	for(int i=0;i<insNum;i ++){
	    		int tmpnum = (int)(insNum*dataWeights[i]*10);
	    		for(int j=0;j<tmpnum;j ++){
	    			tmpdata.add(i);
	    		}
	    	}
	    	
	    	//re-get the dataset and train it
	    	Object[] permutation = tmpdata.toArray();
	    	Random random = new Random(2013);
	        double[][] bagfeatures = new double[insNum][features[0].length];
	        double[] baglabels = new double[insNum]; 
	        for(int i=0;i<insNum;i ++){
	        	int repInd = random.nextInt(permutation.length*10)%insNum;
	        	int select = (int)permutation[repInd];
	
	        	bagfeatures[i] = features[select];
	        	baglabels[i] = labels[select];
	        	//System.out.println(select);
	        }
	       
	        boost[turns] = new ADecisionTree();
	        boost[turns].train(isCategory, bagfeatures, baglabels);
	        
	        double[] result = new double[insNum];
	        for(int i=0;i<insNum;i ++){
	        	result[i] = boost[turns].predict(features[i]);
	        }
	        
	        //get_error
	       List<Object> errorindex = new ArrayList<Object>();
	       for(int i=0;i<insNum;i ++){
	    	   if(result[i] != labels[i]){
	    		   errors[turns] ++;
	    		   errorindex.add(i);
	    	   }
	       }
	       
	       errors[turns] = 1.0*errors[turns] / (1.0*insNum);
	       
	       //get turns classifier weights
	       classifierWeights[turns] = 0.5*Math.log1p(1.0*(1-errors[turns])/(errors[turns]));
	       
	       //get the whole weights
	       double sum = 0;
	       for(int i=0;i<insNum;i ++){
	    	   if(errorindex.indexOf((Object)i) >= 0)
	    		   sum += dataWeights[i]*Math.pow(Math.E, classifierWeights[turns]);
	    	   else
	    		   sum += dataWeights[i]*Math.pow(Math.E, -1.0*classifierWeights[turns]);
	       }
	       
	       
	     //update weights
	       for(int i=0;i<insNum;i ++){
	    	   if(errorindex.indexOf((Object)i) < 0){
	    		   dataWeights[i] = dataWeights[i]*Math.pow(Math.E, -1.0*classifierWeights[turns])/sum;//todo
	    	   }else{
	    		   dataWeights[i] = dataWeights[i]*Math.pow(Math.E, 1.0*classifierWeights[turns])/sum;
	    	   }
	       }
	      
    	}
	      //System.out.println(Arrays.toString(classifierWeights));
    }

    @Override
    public double predict(double[] features) {
    	
    	//double[] weight = new double[k];
    	double[] result = new double[k];
    	
    	for(int i=0;i<k;i ++){
    		//weight[i] = 0.5*Math.log1p(1.0*(1-errors[i])/(errors[i]));
    		result[i] = boost[i].predict(features);
    	}

    	List<Object> label = new ArrayList<Object>();
    	for(int i=0;i<k;i ++){
    		if(label.indexOf((Object)result[i]) < 0){
    			label.add(result[i]);
    		}
    	}
    	
    	double[] prob = new double[label.size()];
    	for(int i=0;i<label.size();i ++){
    		for(int j=0;j<k;j ++){
    			if(result[j] == (double)label.get(i))
    				prob[i] += classifierWeights[j];
    		}
    	}
    	
    	//System.out.println(Arrays.toString(prob));
    	
    	int max=0;
    	for(int i=0;i<label.size();i ++){
    		if(prob[i] > prob[max])
    			max = i;
    	}
        return (double)label.get(max);
    }
}
