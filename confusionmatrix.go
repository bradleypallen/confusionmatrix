// Package confusionmatrix provides a data structure representing the confusion matrix
// for a binary classifier and methods for calculating a range of performance metrics 
// based on statistics captured in the confusion matrix.
package confusionmatrix

import (
	"math"
	"fmt"
	)

type ConfusionMatrix struct {
    TruePositives int
    FalsePositives int
    TrueNegatives int
    FalseNegatives int
}

func (cm *ConfusionMatrix) Update(actual, predicted bool) {
    if actual {
        if predicted {
            cm.TruePositives += 1
        } else {
            cm.FalseNegatives += 1
        }
    } else {
        if predicted {
            cm.FalsePositives += 1
        } else {
            cm. TrueNegatives += 1
        }
    }
}

func (cm *ConfusionMatrix) Positives() int {
    return cm.TruePositives + cm.FalseNegatives
}

func (cm *ConfusionMatrix) Negatives() int {
    return cm.TrueNegatives + cm.FalsePositives
}

func (cm *ConfusionMatrix) Trials() int {
    return cm.Positives() + cm.Negatives()
}

func (cm *ConfusionMatrix) Accuracy() float64 {
    return float64(cm.TruePositives + cm.TrueNegatives) / float64(cm.Trials())
}

func (cm *ConfusionMatrix) Precision() float64 {
    return float64(cm.TruePositives) / float64(cm.TruePositives + cm.FalsePositives)
}

func (cm *ConfusionMatrix) Recall() float64 {
    return float64(cm.TruePositives) / float64(cm.Positives())
}

func (cm *ConfusionMatrix) F(beta float64) float64 {
    betaSqrd, p, r := math.Pow(beta, 2), cm.Precision(), cm.Recall()
    return (1.0 + betaSqrd) * ((p * r) / ((betaSqrd * p) + r))
}

func (cm *ConfusionMatrix) FalsePositiveRate() float64 {
	return float64(cm.FalsePositives) / float64(cm.Negatives())
}

func (cm *ConfusionMatrix) FalseNegativeRate() float64 {
	return float64(cm.FalseNegatives) / float64(cm.Positives())
}

func (cm *ConfusionMatrix) PositivePredictiveValue() float64 {
	return cm.Precision()
}

func (cm *ConfusionMatrix) FalseDiscoveryRate() float64 {
	return float64(cm.FalsePositives) / float64(cm.TruePositives + cm.FalsePositives)
}

func (cm *ConfusionMatrix) NegativePredictiveValue() float64 {
	return float64(cm.TrueNegatives) / float64(cm.TrueNegatives + cm.FalseNegatives)
}

func (cm *ConfusionMatrix) Sensitivity() float64 {
	return cm.Recall()
}

func (cm *ConfusionMatrix) Specificity() float64 {
	return float64(cm.TrueNegatives) / float64(cm.Negatives())
}

func (cm *ConfusionMatrix) BalancedClassificationRate() float64 {
	return cm.Sensitivity() / cm.Specificity()
}

func (cm *ConfusionMatrix) MatthewsCorrelationCoefficient() float64 {
	tp, tn, fp, fn := float64(cm.TruePositives), float64(cm.TrueNegatives), float64(cm.FalsePositives), float64(cm.FalseNegatives)
	return ((tp * tn) - (fp * fn)) / math.Sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
}

func (cm ConfusionMatrix) String() string {
    return fmt.Sprintf("%d trials: TP=%d, FP=%d, FN=%d, TN=%d, acc=%3.2f, P=%3.2f, R=%3.2f, F1=%3.2f",
        cm.Trials(), 
        cm.TruePositives, 
        cm.FalsePositives, 
        cm.FalseNegatives,
        cm.TrueNegatives,
        cm.Accuracy(),
        cm.Precision(),
        cm.Recall(),
        cm.F(1.0))
}
