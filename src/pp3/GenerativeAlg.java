package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import Jama.Matrix;

/**
 * The class implements generative algorithm
 * @author zhaokunxue
 *
 */
public class GenerativeAlg {

	/**
	 * split data by calss
	 * @param data_with_labels given data set
	 * @return a list with 1.data in class 1 and 2.data in class 2
	 */
	public static List<double[][]> splitDataByClass(double[][] data_with_labels) {
		int dimension = data_with_labels[0].length;
		List<double[][]> data_by_class = new ArrayList<double[][]>();
		List<double[]> data_in_c1 = new ArrayList<double[]>();
		List<double[]> data_in_c2 = new ArrayList<double[]>();
		for (double[] data_record : data_with_labels) {
			if ((int) data_record[dimension - 1] == 1) {
				data_in_c1.addAll(Arrays.asList(data_record));
			} else {
				data_in_c2.addAll(Arrays.asList(data_record));
			}
		}
		data_by_class.add(data_in_c1.toArray(new double[][]{}));
		data_by_class.add(data_in_c2.toArray(new double[][]{}));
		return data_by_class;
	}
	
	/**
	 * count the number of data in class1 and class2
	 * @param data_by_class data in their corresponding class
	 * @return a list with 1.number of data in class1 2. number of data in class2
	 */
	public static List<Double> countNs(List<double[][]> data_by_class) {
		List<Double> ns = new ArrayList<Double>();
		double n1 = data_by_class.get(0).length;
		double n2 = data_by_class.get(1).length;
		ns.add(n1);
		ns.add(n2);
		return ns;
	}
	
	/**
	 * calculate mu for each calss
	 * @param data_by_class data in their corresponding class
	 * @return a list with 1. mu for class1 2. mu for class2
	 */
	public static List<Matrix> calculateMus(List<double[][]> data_by_class) {
		double[][] data_in_c1 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(0));
		double[][] data_in_c2 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(1));
		List<Matrix> mus = new ArrayList<Matrix>();
		List<Double> ns = countNs(data_by_class);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		Matrix data_c1_matrix = new Matrix(data_in_c1).transpose();
		Matrix data_c2_martix = new Matrix(data_in_c2).transpose();
		double[][] data_in_c1_t = data_c1_matrix.getArray();
		double[][] data_in_c2_t = data_c2_martix.getArray();
		double[] mu1 = new double[data_in_c1_t.length];
		double[] mu2 = new double[data_in_c2_t.length];
		
		for (int i = 0; i < data_in_c1_t.length; i++) {
			double temp = 0;
			for (int j = 0; j < data_in_c1_t[0].length; j++) {
				temp += data_in_c1_t[i][j];
			}
			mu1[i] = temp/n1;
		}
		
		Matrix mu1_Matrix = new Matrix(mu1, 1).transpose();
		mus.add(mu1_Matrix);
		for (int i = 0; i < data_in_c2_t.length; i++) {
			double temp = 0;
			for (int j = 0; j < data_in_c2_t[0].length; j++) {
				temp += data_in_c2_t[i][j];
			}
			mu2[i] = temp/n2;
		}
		Matrix mu2_Matrix = new Matrix(mu2, 1).transpose();
		mus.add(mu2_Matrix);
		return mus;
	}

	/**
	 * calculate s for each class
	 * @param data_in_class data in their corresponding class
	 * @param mu the corresponding mu
	 * @return s for that class
	 */
	public static Matrix calculateSForClass (double[][] data_in_class, Matrix mu) {
		double n = data_in_class.length;
		int dimension = data_in_class[0].length;
		List<Matrix> data_matrix = new ArrayList<Matrix>();
		for (int i = 0; i < n; i++) {
			Matrix x_i = new Matrix(data_in_class[i], 1).transpose();
			data_matrix.add(x_i);
		}
		Matrix s_for_class = Matrix.identity(dimension, dimension).times(0);
		for (Matrix x : data_matrix) {
			
			s_for_class = s_for_class.plus(x.minus(mu).times(x.minus(mu).transpose()));
		}

		s_for_class = s_for_class.times(1/n);
		return s_for_class;
	}

	/**
	 * calculate s
	 * @param s1 s for class 1
	 * @param s2 s for class 2
	 * @param ns the number of data in class 1 and 2
	 * @return s
	 */
	public static Matrix calculateS (Matrix s1, Matrix s2, List<Double> ns) {
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		double n = n1 + n2;
		Matrix s = s1.times(n1/n).plus(s2.times(n2/n));
		if (s.det() == (double) 0) {
			int dimension = s.getColumnDimension();
			Matrix reg = Matrix.identity(dimension, dimension).times(Math.pow(10, -9));
			s = s.plus(reg);
		}		
		return s;
	}

	/**
	 * calculate w0
	 * @param mu1 mu for class 1
	 * @param mu2 mu for class 2
	 * @param s s 
	 * @param ns number of data in class 1 and 2
	 * @return w0
	 */
	public static double calculateW0 (Matrix mu1, Matrix mu2, Matrix s, List<Double> ns) {
		Matrix s_inverse = ClassificationAlg.ludecompForInvert(s);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		Matrix c1_part = mu1.transpose().times(s_inverse).times(mu1).times(-0.5);
		Matrix c2_part = mu2.transpose().times(s_inverse).times(mu2).times(0.5);
		double w0 = c1_part.get(0, 0) + c2_part.get(0, 0) + Math.log(n1/n2);
		return w0;
	}

	/**
	 * calcualte w
	 * @param mu1 mu for class 1
	 * @param mu2 mu for class 2
	 * @param s s
	 * @return w
	 */
	public static Matrix calculateW (Matrix mu1, Matrix mu2, Matrix s) {
		Matrix diff = mu1.minus(mu2);
		Matrix s_inverse = ClassificationAlg.ludecompForInvert(s);
		Matrix w = s_inverse.times(diff);
		return w;
	}
	
	/**
	 * make predictions using generative method
	 * @param predict_results prediction results
	 * @param current_training_data_with_labels training data with labels
	 * @param testing_data testing data
	 * @param testing_labels testing data labels
	 * @param n training size
	 */
	public static void generativePredict(HashMap<Integer, List<Double>> predict_results, double[][] current_training_data_with_labels, 
			double[][] testing_data, double[] testing_labels, int n){
		List<double[][]> data_by_class = splitDataByClass(current_training_data_with_labels);
		double[][] data_in_c1 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(0));
		double[][] data_in_c2 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(1));
		List<Double> ns = countNs(data_by_class);
		List<Matrix> mus = calculateMus(data_by_class);
		Matrix mu1 = mus.get(0);
		Matrix mu2 = mus.get(1);
		Matrix s1 = calculateSForClass(data_in_c1, mu1);
		Matrix s2 = calculateSForClass(data_in_c2, mu2);
		Matrix s = calculateS (s1, s2, ns);
		double w0 =  calculateW0 (mu1, mu2, s, ns);
		Matrix w = calculateW (mu1, mu2, s);
		double errors = 0;
		for (int j = 0; j < testing_data.length; j++) {
			Matrix t_j = new Matrix(testing_data[j], 1).transpose();
			double a_value = w.transpose().times(t_j).get(0,0) + w0;
			int predict_class = ClassificationAlg.sigmoidPredict(a_value);
			int true_label = (int)testing_labels[j];
			if (predict_class != true_label) {
				errors++;
			}
		}

		double error_rate = errors / (double)(testing_labels.length);
		if (predict_results.containsKey(n)) {
			List<Double> predicts = predict_results.get(n);
			predicts.add(error_rate);
			predict_results.put(n, predicts);
		} else {
			List<Double> predicts = new ArrayList<Double>();
			predicts.add(error_rate);
			predict_results.put(n, predicts);
		}
	}
}