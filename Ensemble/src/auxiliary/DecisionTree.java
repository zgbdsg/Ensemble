package auxiliary;

import java.util.*;
import java.io.*;

/** Decision Tree
 * This class extends Classifier. It uses training data to make a tree, and can 
 * do classification and regression tasks with this tree. It can deal with 
 * missing attribute value during both training and predicting progress.
 * @author wangxd
 */
public class DecisionTree extends Classifier {

	/* Whether it is a classification task or regression task. */
    boolean isClassification;
	
	/* How many samples are used to train this tree. */
	int size;
	
	/* If this is a classification task, it indciates the number of classes. If 
	 * this is a regression task, it equals Inf (You can regard it as a 
	 * classification task with infinity classes.). */
	int divergence;
	
	/* Record the impurity of root node. */
	double root_impurity = 0;
	
	/* The root node of this decision tree. */
	DecisionTreeNode root = null;
	
	/* Types of impurity*/
	public static final int REGRESSION 	= 1;		// for regression
	public static final int GINI 		= 2;		// Gini impurity
	public static final int MISCLASS 	= 3;		// Misclassification impurity
	public static final int ENTROPY 	= 4;		// Entropy impurity

	/* Some parameters that you can tune. */
	public static final double LEAF_NODE_SIZE_FACTOR = 2;//2;
	public static final double MIN_DELTA_IM_CLASS = 0.02;//0.01;
	public static final double MIN_IM_CLASS	 = 0.1;//0.01;
	public static final double MIN_DELTA_IM_REGRE = 0.1;//0.2;//1;
	public static final double MIN_IM_REGRE	 = 0.0;//0.05;//1;
	public static final double NORMAL_STD_RATIO = 3;
	
    public DecisionTree() {
    }

	/** Train procedure.
	 * @param isCategory isCategory[k] indicates whether the kth attribut is 
	 * discrete or continuous, the last attribute is the label.
	 * @param features features[i] is the feature vector of the ith sample.
	 * @param labels labels[i] is the label of he ith sample.
	 */
	@Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
		/* Record the size of the root node. */
		size = labels.length;
		
		/* Classification task or Regression task. */
        isClassification = isCategory[isCategory.length - 1];

		/* If this is a classification task, 'divergence' indciates the number 
		 * of classes. If this is a regression task, 'divergence' equals Inf 
		 * (You can regard is as a classification task with infinity classes).
		 */
		divergence = isClassification ? (unique(labels)).length : Integer.MAX_VALUE;
		
		/* Filter the root node and kick out outliers. */
		ArrayList<Integer> subset = filter(isCategory, features, labels);
		
		/* Make the decision tree. */
		root = makeDecisionTree(isCategory, features, labels, subset);
		
		/* Set the probability. */
		root.prob = 1;
		
//		display("D:\\tree\\");// Display the tree as a file directory structure.
    }

	/*
	* feature is the feature vector of the test sample
	* you need to return the label of test sample
	*/
    @Override
    public double predict(double[] feature) {
		/* Go down from the root node. */
        return predict(feature, root);
    }
	
	/** Predict the label from the specified node. 
	 * @param feature the feature vector.
	 * @param current indicates the current node.
	 * @return the predicted label, a value for the regression task or a 
	 * category for the classification task.
	 */
	private double predict(double[] feature, DecisionTreeNode current) {
		/* If this is a leaf node, then return the label that this node 
		 * represented. */
		if (current.isLeaf)
			return current.label;
		
		/* Store the candidates that you should try after this node. Usually, 
		 * this list has just one member, but when the sample's key attribute 
		 * is NaN, this list includes all direct descendant nodes of the current
		 * node. */
		ArrayList<DecisionTreeNode> candidates = current.rule.apply(feature);

		/* No NaN was encountered, simply go down through the very subnode. */
		if (candidates.size() == 1)
			return predict(feature, candidates.get(0));
		
		/* NaN was encountered. For classification tasks, the most frequent 
		 * predicted category is returned. For regression tasks, the weighted 
		 * mean of predicted labels is returned. */
		double label = 0;
		if (isClassification) {
			double[] probs = new double[candidates.size()];
			double[] labels = new double[candidates.size()];
			double[] categories = unique(labels);
			double[] weights = new double[categories.length];
			for (int i = 0; i < categories.length; i++) {
				// Start to vote.
				weights[i] = 0;
				for (int j = 0; j < labels.length; j++) {
					if (categories[i] == labels[j]) {
						weights[i] += probs[j];
						break;
					}
				}
			}

			label = categories[max(weights)];
		} else {
			// Calculate the weighted mean.
			for (int i = 0; i < candidates.size(); i++) {
				label += candidates.get(i).prob * predict(feature, candidates.get(i));
			}
		}
		return label;
	}

	/* Return the depth from the root node. */
	int depth () {
		return (int)depth(root);
	}
	
	/* Return the depth from this node. */
	double depth (DecisionTreeNode node) {
		if (node == null)
			return 0;
		if (node.isLeaf)
			return 1;

		double[] depths = new double[node.rule.nodes.size()];
		for (int i = 0; i < depths.length; i++)
			depths[i] = depth(node.rule.nodes.get(i));

		return depths[max(depths)] + 1;
	}
	
	/** Display this decision tree as a file directory structure.
	 * @param dirString under which the tree directories are placed.
	 */
	void display (String dirString) {
		/* If the tree is higher than 15, you'd better give up displaying this 
		 * decision tree as a file directory structure (Because you will have 
		 * problems when deleting directories with extremely long names). */
		if (depth() > 15) {
			System.out.println("depth = " + depth());
			return ;
		}
		File dir = new File(dirString);
		
		/* Display this tree from the root node. */
		display(root, "", dir);
	}
	
	/* Display the decision tree */
	void display (DecisionTreeNode node, String rulePrefix, File dir) {
		/* This string will be the directory's name and can show some 
		 * information of this node. */
		String name = 	rulePrefix + ";" +
						"size=" + node.size + ";" + 
						"im=" + node.impurity + ";" + 
						"det=" + node.detImpurity + ";"; 
						//"prob=" + node.prob + ";" + 
						//"isleaf=" + node.isLeaf + ";";

		File nodeFile = new File(dir,name);
		nodeFile.mkdir();
		if (node.rule != null) {
		DecisionRule rule = node.rule;
		if (rule.isCategory) {
			for (int i = 0; i < rule.groups.size(); i++) {
				rulePrefix = "attr[" + rule.index + "]=(";
				for (int j = 0; j < rule.groups.get(i).size(); j++)
					rulePrefix = rulePrefix + rule.groups.get(i).get(j) + " ";
				rulePrefix = rulePrefix + ")";
				display(rule.nodes.get(i), rulePrefix, nodeFile);
			}
		} else {
			rulePrefix = "attr[" + rule.index + "]lt" + rule.split;
			display(rule.nodes.get(0), rulePrefix, nodeFile);
			rulePrefix = "attr[" + rule.index + "]ge" + rule.split;
			display(rule.nodes.get(1), rulePrefix, nodeFile);
		}
		}

	}

	/** Make the decision tree.
	 * @param isCategory isCategory[k] indicates whether the k-th attribut is 
	 * discrete or continuous, the last attribute is the label.
	 * @param features Each row corresponds to one feature.
	 * @param labels Indicate the labels.
	 * @param subset subset[i] indicates that the i-th sample is in this node.
	 * @return The root node of this subtree.
	 */
	DecisionTreeNode makeDecisionTree( boolean[] isCategory,
									   double[][] features,
									   double[] labels,
									   ArrayList<Integer> subset) {
		
		final int type = isClassification ? GINI : REGRESSION;
	
		/* Is this node a leaf ? */
		// TODO
		
		/* New a node. */
		DecisionTreeNode node = new DecisionTreeNode();		
		
		
		/* Step 1. Pick the best attribute and its split point(s). */
		// Some local variables.
		int index = 0;	// The best attribute.
		double[] detimpurities = new double[isCategory.length - 1];
		double[] splits = new double[isCategory.length - 1];
		ArrayList<ArrayList<ArrayList<Double>>> groups = new ArrayList<ArrayList<ArrayList<Double>>>();
		double[] attrs;
		double[] alldets;
		double[] allsplits;
		ArrayList<ArrayList<ArrayList<Double>>> allgroups;
		ArrayList<Integer> effset;
		
		for (int i = 0; i < isCategory.length - 1; i++) {
			attrs = column(features, i);
			effset = kickNaN(attrs, subset);
			groups.add(new ArrayList<ArrayList<Double>>());
			
			if (effset.size() == 0)
				continue;
			
			if (isCategory[i]) {	// This attribute is disctete.
				allgroups = genGroups(attrs, labels, effset);
				alldets = new double[allgroups.size()];
				
				for (int j = 0; j < allgroups.size(); j++) {
					alldets[j] = detimpurity(attrs, labels, partition(attrs, effset, allgroups.get(j)), type);
				}
				
				index = max(alldets);
				groups.set(i,allgroups.get(index));
				detimpurities[i] = alldets[index];
				
			} else {				// This attribute is continuous.
				allsplits = genSplits(attrs, effset);
				alldets = new double[allsplits.length];
				
				for (int j = 0; j < allsplits.length; j++)
					alldets[j] = detimpurity(attrs, labels, partition(attrs, effset, allsplits[j]), type);
				
				index = max(alldets);
				splits[i] = allsplits[index];
				detimpurities[i] = alldets[index];
			}
		}
		
		index = max(detimpurities);		// The best attribute.

		/* Record some information. */
		node.size = subset.size();
		node.impurity = impurity(labels, subset, type);
		node.detImpurity = detimpurities[index];
		if (root_impurity <= 0)
			root_impurity = node.impurity;
		
		/* Carry out pre-pruning. */
		if (prepruning(node)) {
			// Label it as a leaf node.
			node.isLeaf = true;
			node.label = assignLabel(labels, subset);
			return node;
		}
		
		/* Step 2. Make the subtrees. */
		// ?? Pay attention to case #subsets.size() <= 1#.
		attrs = column(features, index);
		effset = kickNaN(attrs, subset);
		ArrayList<ArrayList<Integer>> subsets = isCategory[index] ? 
							partition(attrs, effset, groups.get(index)) :
							partition(attrs, effset, splits[index]);
		ArrayList<DecisionTreeNode> subNodes = new ArrayList<DecisionTreeNode>();

		if (subsets.size() <= 1) {
			// Label it as a leaf node.
			node.isLeaf = true;
			node.label = assignLabel(labels, subset);
			return node;
		}
		
		for (int i = 0; i < subsets.size(); i++) {
			// Make subtrees.
			DecisionTreeNode nextNode = makeDecisionTree(isCategory, features, labels, subsets.get(i));
			nextNode.prob = (double)(subsets.get(i).size()) / (double)(effset.size());
			subNodes.add(nextNode);
		}
		
		// Record the decision rule used by this node.
		node.rule = isCategory[index] ? 
				new DecisionRule(index, subNodes, groups.get(index)) :
				new DecisionRule(index, subNodes, splits[index]);
		
		return node;
	}
	
	
	/** Data filter. Detect outliers and kick out them.	
	 * @param isCategory isCategory[k] indicates whether the kth attribut is 
	 * discrete or continuous, the last attribute is the label.
	 * @param features features[i] is the feature vector of the ith sample.
	 * @param labels labels[i] is the label of he ith sample.
	 * @return the indexes of those samples that are reserved.
	 */
	ArrayList<Integer> filter ( boolean[] isCategory,
								double[][] features,
								double[] labels ) {
		ArrayList<Integer> subset = new ArrayList<Integer>();
		
		if (isClassification) {
			/* For classification, all samples are included. */
			for (int i = 0; i < labels.length; i++)
				subset.add(i);			
		} else {
			/* For regression, samples with unusual labels are treated as 
			 * outliers. 
			 */
			double mean = mean(labels);
			double std = std(labels);
			double ratio;
			for (int i = 0; i < labels.length; i++) {
				ratio = (labels[i] - mean) / std;
				if (ratio <= NORMAL_STD_RATIO)
					subset.add(i);
			}
		}

		return subset;
	}			
	
	/* Generate groups for discrete attribute. */
	ArrayList<ArrayList<ArrayList<Double>>> genGroups ( double[] attrs,
													double[] labels,
													ArrayList<Integer> subset ) {
		return genGroups( pick(attrs, subset), pick(labels, subset) );
	}

	/* Generate groups for discrete attribute. */
	ArrayList<ArrayList<ArrayList<Double>>> genGroups ( double[] attrs,
													double[] labels ) {
		double[] categories = unique(attrs);
		double[] orders = new double[categories.length];
		int[] counts = new int[categories.length];
		
		ArrayList<ArrayList<ArrayList<Double>>> groups = new ArrayList<ArrayList<ArrayList<Double>>>();
		
		if (isClassification) {
			double label = labels[0];
			for (int i = 0; i < attrs.length; i++) {
				for (int j = 0; j < categories.length; j++) {
					if (attrs[i] == categories[j]) {
						orders[j] += (labels[i] == label) ? 1 : 0;
						break;
					}
				}
			}
		
		} else {
			for (int i = 0; i < attrs.length; i++) {
				for (int j = 0; j < categories.length; j++) {
					if (attrs[i] == categories[j]) {
						orders[j] += labels[i];
						counts[j] ++;
						break;
					}
				}
			}
			
			for (int i = 0; i < orders.length; i++)
				orders[i] = orders[i] / counts[i];
		}
		
		/* Sort the 'categories' by 'orders'. The result is stored in 
		 * 'orderedCate'.
		 */
		sort(orders, categories);
		
		for (int i = 0; i < categories.length + 1; i++) {
			groups.add(new ArrayList<ArrayList<Double>>());
			groups.get(i).add(new ArrayList<Double>());	// for left branch
			groups.get(i).add(new ArrayList<Double>());	// for right branch
			
			for (int j = 0; j < i; j++)
				groups.get(i).get(0).add(categories[j]);
			
			for (int j = i; j < categories.length; j++)
				groups.get(i).get(1).add(categories[j]);
		}
		
		return groups;
	}
	
	/** Generate splits for continuous attribute.
	 * @param attrs attrs indicates the attribute values.
	 * @param subset subset indicates which attrs are included.
	 * @return the splits.
	 */
	double[] genSplits ( double[] attrs, ArrayList<Integer> subset ) {
		return genSplits(pick(attrs, subset));
	}
	
	/** Generate splits for continuous attribute.
	 * @param attrs attrs indicates the attribute values.
	 * @return the splits.
	 */
	double[] genSplits ( double[] attrs ) {
		double[] values = unique(attrs);
		double[] splits = new double[values.length + 1];
		if (values.length == 0)
			return new double[0];
		
		splits[0] = values[0];
		for (int i = 1; i < values.length; i++)
			splits[i] = (values[i-1] + values[i]) / 2;
		splits[values.length] = values[values.length-1] + 1;
		
		return splits;
	}
	
	/** Partition for discrete attribute. */
	ArrayList<ArrayList<Integer>> partition ( double[] attrs,
										ArrayList<Integer> subset,
										ArrayList<ArrayList<Double>> groups ) {
		
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		
		for (int i = 0; i < groups.size(); i++)
			result.add(new ArrayList<Integer>());
		
		for (int i = 0; i < subset.size(); i++) {
			for (int j = 0; j < groups.size(); j++) {
				if (groups.get(j).contains(attrs[subset.get(i)]))
					result.get(j).add(subset.get(i));
			}
		}
		
		for (int i = groups.size() - 1; i >= 0; i--) {
			if (result.get(i).size() == 0)
				result.remove(i);
		}

		return result;
	}
/*	
	static void show(ArrayList<ArrayList<Double>> array) {
		System.out.print("{");
		for (int i = 0; i < array.size(); i++) {
			System.out.print("{");
			for (int j = 0; j < array.get(i).size(); j++)
				System.out.print(array.get(i).get(j).toString() + " ");
			System.out.print("},");
		}
		System.out.println("}");
	}
*/
/*
	static void show_int(ArrayList<ArrayList<Integer>> array) {
		System.out.print("{");
		for (int i = 0; i < array.size(); i++) {
			System.out.print("{");
			for (int j = 0; j < array.get(i).size(); j++)
				System.out.print(array.get(i).get(j).toString() + " ");
			System.out.print("},");
		}
		System.out.println("}");
	}
*/
	
	/** Partition for continuous attribute. */
	ArrayList<ArrayList<Integer>> partition ( double[] attrs,
										ArrayList<Integer> subset,
										double split ) {
		
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		result.add(new ArrayList<Integer>());	// for left branch
		result.add(new ArrayList<Integer>());	// for right branch
		
		for (int i = 0; i < subset.size(); i++) {
			if (attrs[subset.get(i)] < split)
				result.get(0).add(subset.get(i));
			else
				result.get(1).add(subset.get(i));
		}
		
		// Prune the right branch if it is empty.
		if (result.get(1).size() == 0)
			result.remove(1);

		// Prune the left branch if it is empty.
		if (result.get(0).size() == 0)
			result.remove(0);
		
		return result;
	}
	
	/** Calculate the impurity for some set.
	 * @param labels Indicate the labels.
	 * @param subset subset[i] indicates that the i-th sample is in this node.
	 * @param type There are four candidates for type, DecisionTree.REGRESSION 
	 * for regression, DecisionTree.GINI for Gini impurity, 
	 * DecisionTree.MISCLASS for misclassification impurity,
	 * and DecisionTree.ENTROPY for entropy impurity.
	 * @return The impurity.
	 */
	double impurity ( double[] labels, ArrayList<Integer> subset, int type ) {
		return impurity( pick(labels, subset), type);
	}
	
	/** Calculate the impurity.
	 * @param labels Indicate the labels.
	 * @param type There are four candidates for type, DecisionTree.REGRESSION 
	 * for regression, DecisionTree.GINI for Gini impurity, 
	 * DecisionTree.MISCLASS for misclassification impurity,
	 * and DecisionTree.ENTROPY for entropy impurity.
	 * @return The impurity.
	 */
	double impurity ( double[] labels, int type ) {
		double impurity = 0;
		double[] categories;
		double[] frequency;
		
		switch (type) {
		case REGRESSION : 
			impurity = var(labels);
			break;

		case GINI :
			categories = unique(labels);
			frequency = frequency(labels, categories);

			/* impurity = 1 - sum_i(P_i^2) */
			impurity = 1;
			for (int i = 0; i < frequency.length; i++)
				impurity -= frequency[i] * frequency[i];
			
			break;

		case MISCLASS : 
			categories = unique(labels);
			frequency = frequency(labels, categories);
			impurity = 1 - frequency[max(frequency)];
			break;

		case ENTROPY : 
			categories = unique(labels);
			frequency = frequency(labels, categories);
			for (int i = 0; i < frequency.length; i++)
				impurity -= frequency[i] * Math.log(frequency[i]) / Math.log(2);

			break;

		default :
			System.out.println("[Err] Unknown type of impurity : " + type);
			break;
		}
		
		return impurity;
	}
	
	/** Calculate the drop in impurity.
	 * @param attrs the attribute values.
	 * @param labels the outcome labels.
	 * @param subsets several subsets of samples.
	 * @param type type indicates the strategy used to calculate impurity. See 
	 * function 'impurity'.
	 * @return the delta impurity or the drop in impurity.
	 */
	double detimpurity ( double[] attrs, 
						 double[] labels,
						 ArrayList<ArrayList<Integer>> subsets,
						 int type ) {
		double det, scale, prob, impurity;
		
		ArrayList<Integer> union = union(subsets);
		
		// There is only one branch.
		if (subsets.size() <= 1)
			return 0;
		
		/* Calculate the impurity of union set. It is necessary to recalculate 
		 * the impurity here, bacause the original node may have some NaN.
		 */
		impurity = impurity( pick(labels, union), type);
		det = impurity;
		scale = 0;
		for (int i = 0; i < subsets.size(); i++) {
			prob = (double)(subsets.get(i).size()) / (double)(union.size());
			det -= prob * impurity(labels, subsets.get(i), type);
			scale -= prob * Math.log(prob) / Math.log(2);
		}
		
		/* You can replace the '1' with scale to use Gain Ratio Strategy, though
		 * I don't use gain ratio due to its preference to unblanced split.*/
		det /= 1; 	// det /= scale;
		
		return det;
	}
	
	/* Assignment of leaf node labels. */
	double assignLabel ( double[] labels, ArrayList<Integer> subset ) {
		return assignLabel(pick(labels,subset));
	}

	/* Assignment of leaf node labels. */
	double assignLabel (double[] labels) {
		if (isClassification) {
			/* Count the label with the highest frequency. */
			double[] categories = unique(labels);
			double[] frequency = frequency(labels, categories);
			return categories[max(frequency)];
		} else {
			/* Calculate the mean value of the labels. */
			return mean(labels);
		}
	}
	
	/* Carry out pre-pruning method. */
	boolean prepruning (DecisionTreeNode node) {
		if (node.size < size * Math.pow(LEAF_NODE_SIZE_FACTOR, -divergence)) {
			return true;
		}
		
		if (isClassification) {
			if ( node.impurity < MIN_IM_CLASS ||
				 node.detImpurity < MIN_DELTA_IM_CLASS ) {
				return true;
			}
		} else {
			if ( (node.impurity / root_impurity) < MIN_IM_REGRE ||
				 (node.detImpurity / node.impurity) < MIN_DELTA_IM_REGRE ) {
				return true;
			}
		}
		
		return false;
	}
	
	
	/* Follows are some static tool functions. */
	
	/* Obtain the unique set from an array. */
	static double[] unique ( double[] labels, ArrayList<Integer> subset ) {
		return unique(pick(labels, subset));
	}

	/* Obtain the unique set from an array. */
	static double[] unique (double[] values) {
		Set<Double> categories = new HashSet<Double>();
		for (int i = 0; i < values.length; i++) {
			if (!Double.isNaN(values[i]))		// Skip NaN
				categories.add(values[i]);
		}
		
		double[] result = new double[categories.size()];
		Double[] result_Double = categories.toArray(new Double[0]);
		for (int i = 0; i < result.length; i++)
			result[i] = result_Double[i];
		
		Arrays.sort(result);		// Ensure that the labels are sorted.
		
		return result;
	}
	
	/* Obtain the frequency of each category. */
	public static double[] frequency (	double[] labels,
										ArrayList<Integer> subset,
										double[] categories ) {
		return frequency( pick(labels, subset), categories );
	}
	
	/* Obtain the frequency of each category. */
	public static double[] frequency (	double[] labels, double[] categories ) {
		// Initial values are all zero.
		double[] frequency = new double[categories.length];
		
		for (int i = 0; i < labels.length; i++) {
			for (int j = 0; j < categories.length; j++) {
				if (labels[i] == categories[j]) {
					frequency[j]++;
					break;
				}
			}
		}
		
		// Normalization
		double size = sum(frequency);
		for (int i = 0; i < frequency.length; i++)
			frequency[i] = frequency[i] / size;
		
		return frequency;
	}
	
	/* Kick out NaN */
	ArrayList<Integer> kickNaN (double[] values, ArrayList<Integer> set ) {
		ArrayList<Integer> subset = new ArrayList<Integer>();
		
		for (int i = 0; i < set.size(); i++)
			if (!Double.isNaN(values[set.get(i)]))
				subset.add(set.get(i));
		
		return subset;
	}
	
	/* Union of some sets. */
	ArrayList<Integer> union (ArrayList<ArrayList<Integer>> sets) {
		Set<Integer> unionSet = new HashSet<Integer>();
		ArrayList<Integer> union = new ArrayList<Integer>();
		
		for (int i = 0; i < sets.size(); i++) {
			for (int j = 0; j < sets.get(i).size(); j++) {
				unionSet.add(sets.get(i).get(j));
			}
		}
		
		for (int value : unionSet)
			union.add(value);
		
		return union;
	}
	
	/* Pick out the subset of array 'values'. */
	static double[] pick ( double[] values, ArrayList<Integer> subset ) {
		double[] result = new double[subset.size()];
		for (int i = 0; i < result.length; i++)
			result[i] = values[subset.get(i)];
		return result;
	}
	
	/* The column vector of a matrix. */
	static double[] column ( double[][] matrix, int c ) {
		if ( matrix.length == 0 || matrix[0].length == 0 || 
			 c < 0 || c >= matrix[0].length) {
			System.out.println("[Err] column : Index out of boundary.");
			return new double[0];
		}
		
		double[] column = new double[matrix.length];
		for (int i = 0; i < column.length; i++)
			column[i] = matrix[i][c];
		
		return column;
	}
	
	/* Return the first index of the maximal value in the array. */
	static int max(double[] values) {
		int index = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > values[index])
				index = i;
		}
		
		if (values.length > 0)
			return index;
		else
			return -1;
	}

	/* Calculate the sum of array 'values'. */
	static double sum (double[] values) {
		double sum = 0;
		for (int i = 0; i < values.length; i++) {
			if (!Double.isNaN(values[i]))
				sum += values[i];
		}
		return sum;
	}
	
	/* Calculate the mean value of array 'values'. */
	static double mean (double[] values) {
		double mean = 0;
		for (int i = 0; i < values.length; i++)
			mean = mean + values[i];
		
		return mean / values.length;
	}
	
	/* Calculate the variance of array 'values'. */
	static double var (double[] values) {
		double mean = mean(values);
		double var = 0;
		
		for (int i = 0; i < values.length; i++)
			var = var + (values[i] - mean) * (values[i] - mean);
		
		return var / values.length;
	}
	
	/* Calculate the standar variance of array 'values'. */
	static double std (double[] values) {
		return Math.sqrt(var(values));
	}
	
	/* Sort the 'values' by 'orders'. The result is stored in 
	 * 'values'.
	 */
	static void sort (double[] orders, double[] values) {
		int index = 0;
		double temp;
		for (int i = 0; i < orders.length; i++) {
			index = i;
			for (int j = i + 1; j < orders.length; j++) {
				if (orders[j] < orders[index]) {
					index = j;
				}
			}
			temp = values[i];
			values[i] = values[index];
			values[index] = temp;
			
			temp = orders[i];
			orders[i] = orders[index];
			orders[index] = temp;
		}
	}
	
	/* Define the class of decision tree node. */
	private class DecisionTreeNode {
		/* Whether this node is a leaf node or not. Default as a inner node.*/
		boolean isLeaf = false;
		
		/* Node size */
		int size;
		
		/* Impurity of this node. */
		double impurity;
		
		/* Drop in impurity after this node. */
		double detImpurity;
		
		/* The decision probability to reach this node from its father. */
		double prob;
		
		/*  If this is a leaf node, it indicates the label of this leaf node. */
		double label;
		
		/* The decision rule. */
		DecisionRule rule = null;
	}
	
	/* Define the class of rule. */
	private class DecisionRule {
		/* The target tree nodes. */
		ArrayList<DecisionTreeNode> nodes;

		/* index indicates which attribute is used. */
		int index;
		
		/* This indicates whether the selected attribute is discrete or not. */
		boolean isCategory;
		
		/* If isCategory is false, split indicates the split point of this 
		 * attribute. */
		double split;
		
		/* If isCatefory is true, this stores groups of categories. */
		ArrayList<ArrayList<Double>> groups;
		
		
		/* Initialize for continuous attribute. */
		public DecisionRule ( int index,
							  ArrayList<DecisionTreeNode> nodes,
							  double split ) {
			this.index = index;
			this.nodes = nodes;
			
			this.isCategory = false;
			this.split = split;
		}
		
		/* Initialize for discrete attribute. */
		public DecisionRule ( int index,
							  ArrayList<DecisionTreeNode> nodes,
							  ArrayList<ArrayList<Double>> groups ) {
			this.index = index;
			this.nodes = nodes;

			this.isCategory = true;
			this.groups = groups;
		}
		
		/* Tell whether this feature has missing value. */
		public boolean hasMissingValue (double []feature) {
			if (Double.isNaN(feature[index]))
				return true;
			else
				return false;
		}
		
		/* Apply this rule to split samples. */
		public ArrayList<DecisionTreeNode> apply (double[] feature) {
			/* Come with missing value. */
			if (hasMissingValue(feature))
				return nodes;
			
			ArrayList<DecisionTreeNode> result = new ArrayList<DecisionTreeNode>();
			if (isCategory) {	// for discrete attribute
				for (int i = 0; i < groups.size(); i++) {
					if (groups.get(i).contains(feature[index]))
						result.add(nodes.get(i));
				}
			} else {
				if (feature[index] < split)
					result.add(nodes.get(0));
				else
					result.add(nodes.get(1));
			}
			
			if (result.size() == 0)
				return nodes;
			else
				return result;
		}
	}
	
}
