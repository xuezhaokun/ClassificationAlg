package pp3;

import java.util.List;

public class ErrorStat {
	public static double calcMean(List<Double> errors) {
		  double sum = 0;
		  if(!errors.isEmpty()) {
		    for (double e : errors) {
		        sum += e;
		    }
		    return sum / errors.size();
		  }
		  return sum;
	}
	
	
	public static double calStD(List<Double> errors) {
        double mean = calcMean(errors);
        double temp = 0;
        for(double e : errors) {
    			temp += (e-mean)*(e-mean);
        }
       double variance = temp / errors.size();
       return Math.sqrt(variance);
	}
}
