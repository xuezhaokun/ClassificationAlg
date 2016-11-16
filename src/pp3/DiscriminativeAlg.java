package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import Jama.Matrix;

/**
 * the class implements discriminative method
 * @author zhaokunxue
 *
 */
public class DiscriminativeAlg {
	
	/**
	 * the function calculates y and R
	 * @param w weighted vector w
	 * @param phi data martix
	 * @return a list with 1:R and 2:y
	 */
	public static List<Matrix> calculateRandY(Matrix w, double[][] phi) {
		List<Matrix> r_y = new ArrayList<Matrix>();
		int data_n = phi.length;
		Matrix r = Matrix.identity(data_n, data_n);
		double[] ys = new double[data_n];
		for (int i = 0; i < data_n; i++) {
			Matrix phi_i = new Matrix(phi[i], 1).transpose(); // d*1
			double a_i = w.transpose().times(phi_i).get(0, 0); // d*1 * 1*d
			double y_i = ClassificationAlg.sigmoid(a_i);
			double r_ii = y_i * (1 - y_i);
			r.set(i, i, r_ii);
			ys[i] = y_i;
		}
		Matrix y_matrix = new Matrix(ys, 1).transpose(); // n*1
		r_y.add(r);
		r_y.add(y_matrix);
		return r_y;
	}
	
	/**
	 * calculate sn
	 * @param phi_matrix data matrix
	 * @param s0_inverse inverse of s0 
	 * @param r R
	 * @return sn
	 */
	public static Matrix calculateSn (Matrix phi_matrix, Matrix s0_inverse, Matrix r) {
		Matrix sn_inverse = phi_matrix.transpose().times(r).times(phi_matrix).plus(s0_inverse);
		return ClassificationAlg.ludecompForInvert(sn_inverse);
	}
	
	/**
	 * calculate sigma square
	 * @param x_n_1 testing data record
	 * @param sn sn
	 * @return sigma square
	 */
	public static double calculateSigSq(Matrix x_n_1, Matrix sn) {
		double sigmaSq = x_n_1.transpose().times(sn).times(x_n_1).get(0, 0);
		return sigmaSq;
	}
	
	/**
	 * calculate mu_a
	 * @param x_n_1 teating data record
	 * @param w_map map for w
	 * @return mu_a
	 */
	public static double calculateMua(Matrix x_n_1, Matrix w_map) {
		double mua = w_map.transpose().times(x_n_1).get(0, 0);
		return mua;
	}
	
	/**
	 * calculate predictive distribution
	 * @param mua mu_a
	 * @param sigSq sigma square
	 * @return predict result
	 */
	public static int predictiveDist (double mua, double sigSq){
		double denominator = Math.sqrt(1 + (Math.PI*sigSq/8));
		double a = mua / denominator;
		int predict = ClassificationAlg.sigmoidPredict(a);
		return predict;
	}
	
	/**
	 * check whether the routine converges
	 * @param newW new w
	 * @param oldW old w
	 * @return true: converged; else false
	 */
	public static boolean checkConverge(Matrix newW, Matrix oldW) {
		double converge_error = Math.pow(10,-3);
		double[] old_array = oldW.getRowPackedCopy();
		double[] new_array = newW.getRowPackedCopy();
		double diff_norm = 0;
		double old_norm = 0;
		for (int i = 0; i < old_array.length; i++) {
			diff_norm += Math.pow((new_array[i] - old_array[i]), 2);
			old_norm += Math.pow(old_array[i], 2); 
		}
		if ((diff_norm / old_norm) < converge_error) {
			return true;
		} else {
			return false;
		}
	}
	
	/**
	 * make prediction using discriminative method for task 1
	 * @param predict_results a hashmap to remember predict results
	 * @param current_training_data_with_labels given training data with labels
	 * @param testing_data testing data
	 * @param testing_labels testing data labels
	 * @param n trianing size
	 */
	public static void discriminativePredict (HashMap<Integer, List<Double>> predict_results, double[][] current_training_data_with_labels, 
			double[][] testing_data, double[] testing_labels, int n) {
		
		boolean converged = false;
		int counter = 0;
		double alpha = 0.1;
		double[][] current_training_data = ClassificationAlg.addW0ToData(ClassificationAlg.getDataFromDataWithLabels(current_training_data_with_labels));
		double[] current_training_labels = ClassificationAlg.getLabelsFromDataWithLabels(current_training_data_with_labels);
		Matrix current_phi = new Matrix(current_training_data);
		int data_n = current_phi.getRowDimension();
		int feature_n = current_phi.getColumnDimension();
		Matrix current_t_matrix = new Matrix(current_training_labels, 1).transpose();
		double[] w = new double[feature_n];
		Arrays.fill(w, 0);
		Matrix w_matrix = new Matrix(w, 1).transpose();
		double[] y = new double[data_n];
		Arrays.fill(y, 0);
		Matrix y_matrix = new Matrix(y, 1).transpose();
		Matrix r = Matrix.identity(data_n, data_n);
		Matrix new_w = null;
		while (!converged && counter < 100) {
			List<Matrix> r_y = calculateRandY(w_matrix, current_training_data);
			r = r_y.get(0);
			y_matrix = r_y.get(1);
			new_w = UpdateMethod.updateNewtonW(current_phi, current_t_matrix, r, y_matrix, w_matrix);
			converged = checkConverge(new_w, w_matrix);
			w_matrix = new_w;
			counter++;
		}
		Matrix identity = Matrix.identity(feature_n, feature_n);
		Matrix s0_inverse = ClassificationAlg.ludecompForInvert(identity.times(1/alpha));
		Matrix sn = calculateSn(current_phi, s0_inverse, r);
		double errors = 0;
		double[][] test_data_with_w0 = ClassificationAlg.addW0ToData(testing_data);
		for (int j = 0; j < test_data_with_w0.length; j++) {
			Matrix test_j = new Matrix(test_data_with_w0[j], 1).transpose();
			double sigSq = calculateSigSq(test_j, sn);
			double mua = calculateMua(test_j, w_matrix);
			int predict_class = predictiveDist (mua, sigSq);
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
	
	/**
	 * update w for task2
	 * @param update_time a list to remember update time for ws
	 * @param training_data_with_w0 training data with w0 added 
	 * @param t_matrix training data label matrix
	 * @param method_option method option 0:newton 1:gradient
	 * @param partialGradient 1:A.csv to take partial w updates(every 30 updates) 0:remember all updates for gradient method
	 * @return a list of ws
	 */
	public static List<Matrix> getUpdateWs(List<Double> update_time, double[][] training_data_with_w0,  
			Matrix t_matrix, int method_option, int partialGradient) {
		boolean converged = false;
		int counter_limit = 100;
		if (method_option == 1) {
			counter_limit = 6000;
		}
		int counter = 0;
		List<Matrix> ws = new ArrayList<Matrix>();
		Matrix phi = new Matrix(training_data_with_w0);
		int data_n = phi.getRowDimension();
		int feature_n = phi.getColumnDimension();
		
		double[] w = new double[feature_n];
		Arrays.fill(w, 0);
		Matrix w_matrix = new Matrix(w, 1).transpose();
		
		double[] y = new double[data_n];
		Arrays.fill(y, 0);
		Matrix y_matrix = new Matrix(y, 1).transpose();
		
		Matrix r = Matrix.identity(data_n, data_n);
		Matrix new_w = null;
		long t_start = System.currentTimeMillis();
		while (!converged && counter < counter_limit) {
			counter++;
			List<Matrix> r_y = calculateRandY(w_matrix, training_data_with_w0);
			r = r_y.get(0);
			y_matrix = r_y.get(1);
			
			if(method_option == 0) {
				new_w = UpdateMethod.updateNewtonW(phi, t_matrix, r, y_matrix, w_matrix);
				long t_end = System.currentTimeMillis();
				long t_delta = t_end - t_start;
				double elapsedSeconds = t_delta / 1000.0;
				update_time.add(elapsedSeconds);
				ws.add(new_w);
				converged = checkConverge(new_w, w_matrix);
				w_matrix = new_w;
			} else {
				new_w = UpdateMethod.updateGradientW(phi, t_matrix, y_matrix, w_matrix);
				if (partialGradient == 1) {
					long t_end = System.currentTimeMillis();
					long t_delta = t_end - t_start;
					double elapsedSeconds = t_delta / 1000.0;
					update_time.add(elapsedSeconds);
					ws.add(new_w);
					converged = checkConverge(new_w, w_matrix);
					w_matrix = new_w;
				} else {
					if (counter%30 == 0) {
						long t_end = System.currentTimeMillis();
						long t_delta = t_end - t_start;
						double elapsedSeconds = t_delta / 1000.0;
						update_time.add(elapsedSeconds);
						ws.add(new_w);	
					}
					converged = checkConverge(new_w, w_matrix);
					w_matrix = new_w;
					if (converged) {
						long t_end = System.currentTimeMillis();
						long t_delta = t_end - t_start;
						double elapsedSeconds = t_delta / 1000.0;
						update_time.add(elapsedSeconds);
						ws.add(new_w);
					}
				}
			}
		}
		return ws;
	}
	
	/**
	 * make predictions with the given w
	 * @param w given w
	 * @param training_data_with_w0 given training dataset with w0 added
	 * @param testing_data testing data
	 * @param testing_labels testing data with labels
	 * @return the prediction result
	 */
	public static double predictForDifferentW(Matrix w, double[][] training_data_with_w0, double[][] testing_data, double[] testing_labels) {
		double alpha = 0.1;
		List<Matrix> r_y = calculateRandY(w, training_data_with_w0);
		Matrix r = r_y.get(0);
		Matrix phi = new Matrix(training_data_with_w0);
		int feature_n = phi.getColumnDimension();
		Matrix identity = Matrix.identity(feature_n, feature_n);
		Matrix s0_inverse = ClassificationAlg.ludecompForInvert(identity.times(1/alpha));
		Matrix sn = calculateSn(phi, s0_inverse, r);	
		double errors = 0;
		double[][] test_data_with_w0 = ClassificationAlg.addW0ToData(testing_data);
		for (int j = 0; j < test_data_with_w0.length; j++) {
			Matrix test_j = new Matrix(test_data_with_w0[j], 1).transpose();
			double sigSq = calculateSigSq(test_j, sn);
			double mua = calculateMua(test_j, w);
			int predict_class = predictiveDist (mua, sigSq);
			int true_label = (int)testing_labels[j];
			if (predict_class != true_label) {
				errors++;
			}
		}
		double error_rate = errors / (double)(testing_labels.length);
		return error_rate;
	}
}
