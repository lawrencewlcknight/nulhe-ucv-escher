package com.ibot;

import java.io.*;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;
import com.biotools.meerkat.Card;
import poker.MaverickGameInfo;
import poker.HandEvaluator;
import poker.ai.HandPotential;

/** Local JSON-lines probability probe. Original classification and action methods
 * are inherited unchanged. Capture stops execution before Math.random(). */
public final class FellOmenBridge extends FellOmen_2_Impl implements AutoCloseable {
    private static final class Captured extends RuntimeException {
        final String table; final double[][] values;
        Captured(String t, double[][] v) { super(null,null,false,false); table=t; values=v; }
    }
    private final Map<String, ZipFile> index = new HashMap<>();
    private final List<ZipFile> archives = new ArrayList<>();
    private final Map<String, double[][]> tables = lru(128);
    private final Map<String, double[]> features = lru(128);
    private final Map<String, Integer> flops = lru(128);
    private boolean reference;
    private static <T> Map<String,T> lru(final int size) {
        return new LinkedHashMap<String,T>(size, .75f, true) {
            protected boolean removeEldestEntry(Map.Entry<String,T> e) { return size()>size; }
        };
    }
    public FellOmenBridge() throws IOException {
        for (int k=1;k<=7;k++) {
            ZipFile zip = new ZipFile("Strategy_FellOmen_2/"+k+".zip");
            archives.add(zip);
            Enumeration<? extends ZipEntry> entries=zip.entries();
            while(entries.hasMoreElements()) {
                ZipEntry e=entries.nextElement();
                if(!e.isDirectory()) {
                    String name=e.getName().substring(e.getName().lastIndexOf('/')+1);
                    // The original reader searches archives 1..7 and returns
                    // the first match; several filenames occur more than once.
                    index.putIfAbsent(name,zip);
                }
            }
        }
    }
    @Override public boolean getDebug() { return false; }
    @Override public void debug(String message) {}
    @Override protected double[][] getActionTable(int round, String file) {
        if(!index.containsKey(file)) throw new IllegalArgumentException("Missing strategy: "+file);
        double[][] values;
        if(reference) {
            // Independent original ZIP scanner and matrix parser for parity checks.
            values=super.getActionTable(round,file);
            if(values==null) throw new IllegalStateException("Original table reader failed: "+file);
        } else {
            values=tables.get(file);
            if(values==null) {
                int size=new int[]{169,177,131,75}[round];
                values=new double[3][size];
                ZipFile zip=index.get(file);
                String prefix=new File(zip.getName()).getName().replace(".zip","");
                try(InputStream in=zip.getInputStream(zip.getEntry(prefix+"/"+file));
                    Scanner scanner=new Scanner(in,StandardCharsets.UTF_8.name()).useLocale(Locale.US)) {
                    for(int row=2;row>=0;row--) for(int col=0;col<size;col++) values[row][col]=scanner.nextDouble();
                    if(scanner.hasNext()) throw new IllegalStateException("Extra table data: "+file);
                } catch(IOException e) { throw new UncheckedIOException(e); }
                tables.put(file,values);
            }
        }
        throw new Captured(file,values);
    }
    private double[] classify(String hole, String board, int stage) {
        String key=hole+"|"+board;
        if(!reference && features.containsKey(key)) return features.get(key);
        String[] cards=hole.split(" ");
        double hs=new HandEvaluator().handRank(cards[0],cards[1],board);
        double pp=0, np=0;
        if(stage==1 || stage==2) {
            HandPotential potential=new HandPotential();
            pp=potential.ppot_raw(cards[0],cards[1],board,true);
            np=potential.getLastNPot();
        }
        double[] result={hs,pp,np};
        features.put(key,result);
        return result;
    }
    private static MaverickGameInfo gameInfo(final int stage, final int actor) {
        return (MaverickGameInfo)Proxy.newProxyInstance(
            MaverickGameInfo.class.getClassLoader(),new Class<?>[]{MaverickGameInfo.class},
            (proxy,method,args)-> {
                switch(method.getName()) {
                    case "getStage": return stage;
                    case "getCurrentPlayerSeat": return actor;
                    case "getButtonSeat": return 0;
                    default: throw new UnsupportedOperationException(method.getName());
                }
            });
    }
    private static void verifyHistory(Map<String,String> q, int stage, int actor, int raises) {
        String[] rounds=q.get("history").split("/",-1);
        if(rounds.length!=stage+1) throw new IllegalArgumentException("Street/history mismatch");
        ReferenceHistory ref=new ReferenceHistory();
        String[] keys={"preflop","flop","turn"};
        for(int st=0;st<=stage;st++) {
            ref.newRound(st);
            int who=st==0?0:1, count=0, terminal=-1;
            for(char action:rounds[st].toCharArray()) {
                if(action=='r') { ref.raise(); count++; }
                else if(action=='c') terminal=ref.nodeToTerminalNodeIndex(ref.get_action_node(gameInfo(st,who)));
                else throw new IllegalArgumentException("Invalid public action");
                who=1-who;
            }
            if(st<stage && terminal!=Integer.parseInt(q.get(keys[st])))
                throw new IllegalArgumentException("Terminal history differs from original adapter");
            if(st==stage && (count!=raises || who!=actor))
                throw new IllegalArgumentException("Current action history mismatch");
        }
    }
    private String query(Map<String,String> q) {
        int stage=Integer.parseInt(q.get("street")), actor=Integer.parseInt(q.get("actor"));
        int raises=Integer.parseInt(q.get("raises"));
        if(stage<0 || stage>3 || actor<0 || actor>1 || raises<0 || raises>(stage==0?3:4))
            throw new IllegalArgumentException("Invalid stage, actor or raise count");
        verifyHistory(q,stage,actor,raises);
        String hole=q.get("hole"), board=q.get("board");
        String[] privateCards=hole.split(" ");
        if(privateCards.length!=2) throw new IllegalArgumentException("Two private cards required");
        String[] publicCards=board.isEmpty()?new String[0]:board.split(" ");
        if(publicCards.length!=new int[]{0,3,4,5}[stage]) throw new IllegalArgumentException("Invalid board size");
        Set<String> seen=new HashSet<>();
        for(String card:(hole+(board.isEmpty()?"":" "+board)).split(" "))
            if(!card.matches("[2-9TJQKA][cdhs]") || !seen.add(card)) throw new IllegalArgumentException("Invalid or duplicate card");
        reference="reference".equals(q.get("mode"));
        MaverickGameInfo gi=gameInfo(stage,actor);
        newRound(stage);
        for(int i=0;i<raises;i++) raise();
        Card c1=new Card(privateCards[0]), c2=new Card(privateCards[1]);
        int bucket, flop=-1;
        double[] f={0,0,0};
        if(stage==0) bucket=getPreflopIndex(c1,c2);
        else {
            String flopString=String.join(" ",Arrays.copyOf(publicCards,3));
            if(reference || !flops.containsKey(flopString)) flops.put(flopString,getFlopBoardIndex(flopString));
            flop=flops.get(flopString);
            f=classify(hole,board,stage);
            double effective=f[0]*(1-f[2])+(1-f[0])*f[1];
            bucket=stage==1?getFlopIndex(effective,f[1]):stage==2?getTurnIndex(effective,f[1]):getRiverIndex(f[0]);
        }
        if(bucket<0) throw new IllegalStateException("Invalid hand bucket");
        int pre=Integer.parseInt(q.get("preflop")), fl=Integer.parseInt(q.get("flop")), turn=Integer.parseInt(q.get("turn"));
        for(int i:new int[]{pre,fl,turn}) if(i < -1 || i>8) throw new IllegalArgumentException("Invalid round history");
        int node=get_action_node(gi);
        try {
            if(stage==0) preFlopAction(gi,c1,c2,actor);
            else if(stage==1) flopAction(gi,f[0],f[1],f[2],flop,actor,pre);
            else if(stage==2) turnAction(gi,f[0],f[1],f[2],flop,actor,pre,fl);
            else riverAction(gi,f[0],flop,actor,pre,fl,turn);
        } catch(Captured c) {
            return "{\"probabilities\":["+c.values[0][bucket]+","+c.values[1][bucket]+","+c.values[2][bucket]+
                "],\"table\":\""+c.table+"\",\"bucket\":"+bucket+",\"flop_bucket\":"+flop+
                ",\"node\":"+node+",\"hand_strength\":"+f[0]+",\"positive_potential\":"+f[1]+",\"negative_potential\":"+f[2]+"}";
        }
        throw new IllegalStateException("Probability probe did not capture a table");
    }
    /** Deliberately small JSON object parser: protocol values are strings/integers. */
    private static Map<String,String> parse(String line) throws IOException {
        StreamTokenizer t=new StreamTokenizer(new StringReader(line));
        t.resetSyntax(); t.whitespaceChars(0,32); t.quoteChar('"'); t.wordChars('0','9'); t.wordChars('-','-');
        Map<String,String> result=new HashMap<>();
        if(t.nextToken()!='{') throw new IOException("Expected object");
        while(t.nextToken()!='}') {
            if(t.ttype!='"') throw new IOException("Expected key");
            String key=t.sval;
            if(t.nextToken()!=':') throw new IOException("Expected colon");
            if(t.nextToken()!='"' && t.ttype!=StreamTokenizer.TT_WORD) throw new IOException("Expected string/integer");
            if(result.put(key,t.sval)!=null) throw new IOException("Duplicate key");
            int next=t.nextToken();
            if(next=='}') break;
            if(next!=',') throw new IOException("Expected comma");
        }
        if(t.nextToken()!=StreamTokenizer.TT_EOF) throw new IOException("Trailing data");
        return result;
    }
    @Override public void close() throws IOException { for(ZipFile z:archives) z.close(); }
    public static void main(String[] args) throws Exception {
        try(FellOmenBridge bridge=new FellOmenBridge();
            BufferedReader in=new BufferedReader(new InputStreamReader(System.in,StandardCharsets.UTF_8))) {
            String line;
            while((line=in.readLine())!=null) {
                try { System.out.println(bridge.query(parse(line))); }
                catch(Exception e) {
                    String message=(e.getClass().getSimpleName()+": "+e.getMessage()).replace("\\","\\\\").replace("\"","\\\"").replace("\n"," ").replace("\r"," ");
                    System.out.println("{\"error\":\""+message+"\"}");
                }
                System.out.flush();
            }
        }
    }
}
