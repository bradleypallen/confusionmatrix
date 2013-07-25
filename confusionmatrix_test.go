package confusionmatrix

import "testing"

func TestConfusionMatrix (t *testing.T) {
	const tp, fn, fp, tn = 293, 7, 8, 292
	cm := ConfusionMatrix{tp, fp, tn, fn}
	// Positives()
	if pos := cm.Positives(); pos != tp + fn {
		t.Errorf("Positives() = %v, want %v", pos, tp + fn)
	}
	// Negatives()
	if neg := cm.Negatives(); neg != tn + fp {
		t.Errorf("Negatives() = %v, want %v", neg, tn + fp)
	}
	// Trials()
	if tri := cm.Trials(); tri != tp + fn + tn + fp {
		t.Errorf("Negatives() = %v, want %v", tri, tp + fn + tn + fp)
	}
	// Accuracy()
	if acc := cm.Accuracy(); acc != float64(tp + tn) / float64(tp + fn + tn + fp) {
		t.Errorf("Accuracy() = %v, want %v", acc, float64(tp + tn) / float64(tp + fn + tn + fp))
	}
	// Precision()
	if pre := cm.Precision(); pre != float64(tp) / float64(tp + fp) {
		t.Errorf("Precision() = %v, want %v", pre, float64(tp) / float64(tp + fp))
	}
	// Recall()
	if rec := cm.Recall(); rec != float64(tp) / float64(tp + fn) {
		t.Errorf("Recall() = %v, want %v", rec, float64(tp) / float64(tp + fn))
	}
	// F(beta float64)
	if f := cm.F(1.0); f != 0.9750415973377703 {
		t.Errorf("F(1.0) = %v, want %v", f, 0.9750415973377703)
	}
	// FalsePositiveRate()
	if fpr := cm.FalsePositiveRate(); fpr != float64(fp) / float64(tn + fp) {
		t.Errorf("FalsePositiveRate() = %v, want %v", fpr, float64(fp) / float64(tn + fp))
	}
	// FalseNegativeRate()
	if fnr := cm.FalseNegativeRate(); fnr != float64(fn) / float64(fn + tp) {
		t.Errorf("FalseNegativeRate() = %v, want %v", fnr, float64(fn) / float64(fn + tp))
	}
	// PositivePredictiveValue()
	if ppv := cm.PositivePredictiveValue(); ppv != float64(tp) / float64(tp + fp) {
		t.Errorf("PositivePredictiveValue() = %v, want %v", ppv, float64(tp) / float64(tp + fp))
	}
	// FalseDiscoveryRate()
	if fdr := cm.FalseDiscoveryRate(); fdr != 0.026578073089700997 {
		t.Errorf("FalseDiscoveryRate() = %v, want %v", fdr, 0.026578073089700997)
	}
	// NegativePredictiveValue()
	if npv := cm.NegativePredictiveValue(); npv != 0.9765886287625418 {
		t.Errorf("NegativePredictiveValue() = %v, want %v", npv, 0.9765886287625418)
	}
	// Sensitivity()
	if sen := cm.Sensitivity(); sen != float64(tp) / float64(tp + fn) {
		t.Errorf("Sensitivity() = %v, want %v", sen, float64(tp) / float64(tp + fn))
	}
	// Specificity()
	if spec := cm.Specificity(); spec != float64(tn) / float64(tn + fp) {
		t.Errorf("Specificity() = %v, want %v", spec, float64(tn) / float64(tn + fp))
	}
	// BalancedClassificationRate()
	if bcr := cm.BalancedClassificationRate(); bcr != 1.0034246575342465 {
		t.Errorf("BalancedClassificationRate() = %v, want %v", bcr, 1.0034246575342465)
	}
	// MatthewsCorrelationCoefficient()
	if mcc := cm.MatthewsCorrelationCoefficient(); mcc != 0.9500052778217596 {
		t.Errorf("MatthewsCorrelationCoefficient() = %v, want %v", mcc, 0.9500052778217596)
	}
}
