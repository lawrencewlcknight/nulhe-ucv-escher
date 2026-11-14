package poker.ai.model;
/** FO2 uses ppot_raw, never the weighted opponent-model path. */
public final class WeightTable {
 public double getCell(int a, int b) { throw new UnsupportedOperationException("Weighted potential is not part of FO2"); }
}
