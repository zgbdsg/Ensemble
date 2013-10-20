/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package auxiliary;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author zhouguobing
 */
public class RandomForest extends Classifier {

	int Treenum;
	DecisionTree[] forest;
	
    public RandomForest() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
    	Treenum = 5;
    	forest = new DecisionTree[Treenum];
    	
    	Random random = new Random(2013);
        int[] permutation = new int[features.length];
        for (int i = 0; i < permutation.length; i++) {
            permutation[i] = i;
        }
        for (int i = 0; i < Treenum * permutation.length; i++) {
            int repInd = random.nextInt(permutation.length);
            int ind = i % permutation.length;

            int tmp = permutation[ind];
            permutation[ind] = permutation[repInd];
            permutation[repInd] = tmp;
        }
        
        int length = features.length/5;
        double[][] bagfeatures = new double[length][features[0].clone().length];
        double[] baglabels = new double[length];
        
        for(int i=0;i<Treenum;i ++){
        	forest[i] = new DecisionTree();
        }
        
        for(int i=0;i<Treenum;i ++) {
        	for(int j=0;j<length;j ++) {
        		bagfeatures[j] = features[i*length+j];
        		baglabels[j] = labels[i*length+j];
        	}
        }
        
        for(int i=0;i<Treenum;i ++){
        	forest[i].train(isCategory, bagfeatures, baglabels);
        }
    }

    @Override
    public double predict(double[] features) {
    	List<Object> result = new ArrayList<Object>();
    	double[] predict = new double[Treenum];
    	
    	
        return 0;
    }
}
