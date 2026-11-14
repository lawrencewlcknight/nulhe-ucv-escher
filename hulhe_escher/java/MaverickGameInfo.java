package poker;
/** Minimal type facade; a proxy supplies the getters used by the opponent. */
public interface MaverickGameInfo extends com.biotools.meerkat.GameInfo {
 double getBetAmount();
 double getBetSize();
}
