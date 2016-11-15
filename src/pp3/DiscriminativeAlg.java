package pp3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import Jama.Matrix;
public class DiscriminativeAlg {
	
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
	
	public static Matrix calculateSn (Matrix phi_matrix, Matrix s0_inverse, Matrix r) {
		Matrix sn_inverse = phi_matrix.transpose().times(r).times(phi_matrix).plus(s0_inverse);
		return ClassificationAlg.ludecompForInvert(sn_inverse);
	}
	
	public static double calculateSigSq(Matrix x_n_1, Matrix sn) {
		double sigmaSq = x_n_1.transpose().times(sn).times(x_n_1).get(0, 0);
		return sigmaSq;
	}
	
	public static double calculateMua(Matrix x_n_1, Matrix w_map) {
		double mua = w_map.transpose().times(x_n_1).get(0, 0);
		return mua;
	}
	
	public static int predictiveDist (double mua, double sigSq){
		double denominator = Math.sqrt(1 + (Math.PI*sigSq/8));
		double a = mua / denominator;
		int predict = ClassificationAlg.sigmoidPredict(a);
		return predict;
	}
	
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
			System.out.println((diff_norm / old_norm));
			return true;
		} else {
			return false;
		}
	}
	
	public static double[][] addW0ToData(double[][] dataset){
		int dimension = dataset[0].length;
		int data_length = dataset.length;
		double[][] data_with_w0 = new double[data_length][dimension + 1];
		for (int i = 0; i < data_length; i++) {
			for (int j = 0; j < dimension; j++) {
				data_with_w0[i][j] = dataset[i][j];
			}
			data_with_w0[i][dimension] = (double) 1;
		}
		return data_with_w0;
	}
	
	public static void discriminativePredict (HashMap<Integer, List<Double>> predict_results, double[][] current_training_data_with_labels, 
			double[][] testing_data, double[] testing_labels, int n) {
		
		boolean converged = false;
		int counter = 0;
		double alpha = 0.1;
		double[][] current_training_data = addW0ToData(ClassificationAlg.getDataFromDataWithLabels(current_training_data_with_labels));
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
		double[][] test_data_with_w0 = addW0ToData(testing_data);
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
		System.out.println(error_rate);
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
	
	public static void getUpdateWs(List<Matrix> ws, List<Double> update_time, double[][] training_data,  
			Matrix t_matrix, int method_option) {
		boolean converged = false;
		int counter_limit = 100;
		if (method_option == 1) {
			counter_limit = 6000;
		}
		int counter = 0;
		
		double[][] training_data_with_w0 = addW0ToData(training_data);
		Matrix phi = new Matrix(training_data_with_w0);
		int data_n = phi.getRowDimension();
		int feature_n = phi.getColumnDimension();
		
		double[] w = new double[feature_n];
		Arrays.fill(w, 0);
		Matrix w0 = new Matrix(w, 1).transpose();
		
		double[] y = new double[data_n];
		Arrays.fill(y, 0);
		Matrix y_matrix = new Matrix(y, 1).transpose();
		
		Matrix r = Matrix.identity(data_n, data_n);

		long t_start = System.currentTimeMillis();
		while (!converged && counter < counter_limit) {
			List<Matrix> r_y = calculateRandY(w0, training_data_with_w0);
			r = r_y.get(0);
			y_matrix = r_y.get(1);
		
			Matrix new_w = UpdateMethod.updateGradientW(phi, t_matrix, y_matrix, w0);
			long t_end = System.currentTimeMillis();
			long t_delta = t_end - t_start;
			double elapsedSeconds = t_delta / 1000.0;
			update_time.add(elapsedSeconds);
			ws.add(new_w);
			converged = checkConverge(new_w, w0);
			w0 = new_w;
			counter++;
		}
		System.out.println(counter);
		System.out.println(Arrays.deepToString(w0.getArray()));
	}
	
//	public static void diffUpdateMethodPredict(List<List<Double>> method_data, double[][] training_data_with_labels, 
//			double[][] testing_data, double[] testing_labels, int training_size, int method_option) {
//		boolean converged = false;
//		int counter_limit = 100;
//		if (method_option == 1) {
//			counter_limit = 6000;
//		}
//		int counter = 0;
//		double alpha = 0.1;
//		List<Double> running_time = new ArrayList<Double>();
//		List<Double> error_rates = new ArrayList<Double>();
//		
//		double[][] current_training_data = addW0ToData(ClassificationAlg.getDataFromDataWithLabels(training_data_with_labels));
//		double[] current_training_labels = ClassificationAlg.getLabelsFromDataWithLabels(training_data_with_labels);
//		Matrix current_phi = new Matrix(current_training_data);
//		int data_n = current_phi.getRowDimension();
//		int feature_n = current_phi.getColumnDimension();
//		Matrix current_t_matrix = new Matrix(current_training_labels, 1).transpose();
//		double[] w = new double[feature_n];
//		Arrays.fill(w, 0);
//		Matrix w_matrix = new Matrix(w, 1).transpose();
//		double[] y = new double[data_n];
//		Arrays.fill(y, 0);
//		Matrix y_matrix = new Matrix(y, 1).transpose();
//		Matrix r = Matrix.identity(data_n, data_n);
//		Matrix new_w = null;
//		Matrix identity = Matrix.identity(feature_n, feature_n);
//		Matrix s0_inverse = ClassificationAlg.ludecompForInvert(identity.times(1/alpha));
//		
//		while (!converged && counter < counter_limit) {
//			List<Matrix> r_y = calculateRandY(w_matrix, current_training_data);
//			r = r_y.get(0);
//			y_matrix = r_y.get(1);
//			long t_start = System.currentTimeMillis();
//			if(method_option == 0) {
//				new_w = UpdateMethod.updateNewtonW(current_phi, current_t_matrix, r, y_matrix, w_matrix);
//			} else {
//				new_w = UpdateMethod.updateGradientW(current_phi, current_t_matrix, y_matrix, w_matrix);
//			}
//			long t_end = System.currentTimeMillis();
//			long t_delta = t_end - t_start;
//			double elapsedSeconds = t_delta / 1000.0;
//			running_time.add(elapsedSeconds);
//			converged = checkConverge(new_w, w_matrix);
//			w_matrix = new_w;
//			counter++;
//			//System.out.println(counter);
//			//if (method_option == 1 && counter%60 == 0){
//				
//			//}
//		}
//		Matrix sn = calculateSn(current_phi, s0_inverse, r);	
//		double errors = 0;
//		double[][] test_data_with_w0 = addW0ToData(testing_data);
//		for (int j = 0; j < test_data_with_w0.length; j++) {
//			Matrix test_j = new Matrix(test_data_with_w0[j], 1).transpose();
//			double sigSq = calculateSigSq(test_j, sn);
//			double mua = calculateMua(test_j, w_matrix);
//			int predict_class = predictiveDist (mua, sigSq);
//			int true_label = (int)testing_labels[j];
//			if (predict_class != true_label) {
//				errors++;
//			}
//		}
//		double error_rate = errors / (double)(testing_labels.length);
//		//error_rates.add(error_rate);
//		System.out.println("error: " + error_rate);
//		method_data.add(running_time);
//		method_data.add(error_rates);
//	}
}
