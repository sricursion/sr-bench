from fractions import Fraction
import math, random, itertools, json, pandas as pd
from functools import lru_cache

rng = random.Random(28032026)

def fmt(x):
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return fmt(Fraction(x).limit_denominator())
    raise TypeError(type(x))


def mean_f(vals):
    return Fraction(sum(vals), len(vals))


def var_pop(vals):
    m=mean_f(vals)
    return sum((Fraction(v)-m)**2 for v in vals)/len(vals)


def choose_distinct_ints(n, lo, hi):
    vals=set()
    while len(vals)<n:
        vals.add(rng.randint(lo,hi))
    return sorted(vals)


def divisors(n):
    ds=set()
    for d in range(1, int(math.isqrt(n))+1):
        if n%d==0:
            ds.add(d); ds.add(n//d)
    return sorted(ds)


def prime_factors_distinct(n):
    x=n
    out=[]
    p=2
    while p*p<=x:
        if x%p==0:
            out.append(p)
            while x%p==0:
                x//=p
        p += 1 if p==2 else 2
    if x>1:
        out.append(x)
    return out


def derangement(n):
    if n==0: return 1
    d=[0]*(n+1)
    d[0]=1
    if n>=1: d[1]=0
    for i in range(2,n+1):
        d[i]=(i-1)*(d[i-1]+d[i-2])
    return d[n]


def block_partitions_min2(n,k):
    dp=[[0]*(k+2) for _ in range(n+2)]
    dp[0][0]=1
    for i in range(1,n+1):
        for j in range(1,min(k,i)+1):
            dp[i][j]+=j*dp[i-1][j]
            if i>=2:
                dp[i][j]+=(i-1)*dp[i-2][j-1]
    return dp[n][k]


def count_gap_subset_sum(n,k,S):
    arr=list(range(1,n+1))
    cnt=0
    # DP with previous selected?
    from functools import lru_cache
    @lru_cache(None)
    def dp(pos, prev_taken, kleft, sleft):
        if kleft==0:
            return 1 if sleft==0 else 0
        if pos>n or sleft<0:
            return 0
        # lower/upper pruning
        # choose
        ans=dp(pos+1, 0, kleft, sleft)
        if not prev_taken:
            ans += dp(pos+1, 1, kleft-1, sleft-pos)
        return ans
    return dp(1,0,k,S)


def count_shortest_paths_exact_turns(a,b,t):
    # exact number of E/N strings with a E and b N and exactly t turns
    if a==0 or b==0:
        return 1 if t==0 else 0
    total=0
    # start E or N
    for start in [0,1]:  # 0 E, 1 N
        runs=t+1
        e_runs=(runs+ (1-start))//2 if start==0 else runs//2
        n_runs=runs-e_runs
        if start==1:
            n_runs=(runs+1)//2
            e_runs=runs-n_runs
        # compositions of a into e_runs positive parts, b into n_runs positive parts
        if e_runs>=1 and n_runs>=1 and a>=e_runs and b>=n_runs:
            total += math.comb(a-1, e_runs-1)*math.comb(b-1, n_runs-1)
    return total


def count_walks(adj, start, end, L):
    nodes=sorted(adj)
    idx={u:i for i,u in enumerate(nodes)}
    dp=[0]*len(nodes)
    dp[idx[start]]=1
    for _ in range(L):
        ndp=[0]*len(nodes)
        for u in nodes:
            for v in adj[u]:
                ndp[idx[v]] += dp[idx[u]]
        dp=ndp
    return dp[idx[end]]


def polygon_double_area(coords):
    s=0
    for (x1,y1),(x2,y2) in zip(coords, coords[1:]+coords[:1]):
        s += x1*y2 - x2*y1
    return abs(s)


def line_intersection_sum(p1,p2,q1,q2):
    x1,y1=p1; x2,y2=p2; x3,y3=q1; x4,y4=q2
    # using determinants
    den=Fraction((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4),1)
    px=Fraction((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4), den)
    py=Fraction((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4), den)
    return px+py


def ballot_prefix_count(a,b):
    if a<=b:
        return 0
    return (a-b)*math.comb(a+b,a)//(a+b)


def guild_oath_count(n,k):
    # onto labeled k groups, elder guild at least 2
    return math.factorial(k)*stirling2(n,k) - n*math.factorial(k-1)*stirling2(n-1,k-1)


def stirling2(n,k):
    dp=[[0]*(k+2) for _ in range(n+2)]
    dp[0][0]=1
    for i in range(1,n+1):
        for j in range(1,min(i,k)+1):
            dp[i][j]=j*dp[i-1][j]+dp[i-1][j-1]
    return dp[n][k]


def count_layered_box(n,k,g):
    rem=n-g*(k-1)
    return 0 if rem<k else math.comb(rem,k)


def count_remainder_street(N, conds):
    cnt=0
    for x in range(1,N+1):
        if all(x%m==r for m,r in conds):
            cnt+=1
    return cnt


def count_interval_coprime(L,R,m):
    return sum(1 for x in range(L,R+1) if math.gcd(x,m)==1)


def count_exact_gcd(N,M,d):
    return sum(1 for x in range(1,N+1) if math.gcd(x,M)==d)


def count_quadratic_residue(N,a,m):
    a%=m
    return sum(1 for x in range(1,N+1) if (x*x-a)%m==0)


def expected_conditional_reward(r,b,Rv,Bv,d):
    from math import comb
    total=Fraction(0,1); prob=Fraction(0,1)
    for xr in range(max(0,d-b), min(d,r)+1):
        ways=Fraction(math.comb(r,xr)*math.comb(b,d-xr), math.comb(r+b,d))
        if xr>=1:
            prob += ways
            total += ways * (xr*Rv + (d-xr)*Bv)
    return total/prob


def expected_max_of_two(values):
    # distinct values list or multiset list, draw 2 without replacement uniformly from positions
    n=len(values)
    total=0
    cnt=0
    for i in range(n):
        for j in range(i+1,n):
            total += max(values[i], values[j])
            cnt += 1
    return Fraction(total, cnt)


def triangle_area16(a,b,c):
    s=(a+b+c)/2
    return int((a+b+c)*( -a+b+c)*(a-b+c)*(a+b-c))  # 16A^2 = (a+b+c)(-a+b+c)(a-b+c)(a+b-c)


def silent_candidates(known, n, mean, var, lo=0, hi=40):
    # known sorted with two blanks omitted
    target_sum = mean*n - sum(known)
    target_ss = (var + mean*mean)*n - sum(k*k for k in known)
    cands=[]
    for a in range(lo,hi+1):
        for b in range(a+1,hi+1):
            if a+b==target_sum and a*a+b*b==target_ss:
                allv=sorted(known+[a,b])
                # ensure the known list is subsequence preserving order
                if allv==sorted(allv):
                    cands.append((a,b))
    return cands


def moment_swap_candidates(n,m1,v1,m2,v2,lo=0,hi=60):
    S1 = m1*n
    SS1 = (v1 + m1*m1)*n
    S2 = m2*n
    SS2 = (v2 + m2*m2)*n
    cands=[]
    dS = S2-S1
    dSS = SS2-SS1
    for a in range(lo,hi+1):
        b = a + dS
        if b==int(b): b=int(b)
        if isinstance(b,Fraction) and b.denominator!=1: continue
        if not (lo<=b<=hi): continue
        if b>a and b*b-a*a==dSS:
            cands.append((a,b))
    return cands


def posterior_house_A(pA,a_red,a_blue,b_red,b_blue,obs):
    # obs tuple colors like ('R','B')
    def prob_house(r,b):
        total=r+b
        remr, remb = r,b
        p=Fraction(1,1)
        for c in obs:
            if c=='R':
                p*=Fraction(remr, remr+remb); remr-=1
            else:
                p*=Fraction(remb, remr+remb); remb-=1
        return p
    pA=Fraction(pA)
    pB=1-pA
    LA=prob_house(a_red,a_blue)
    LB=prob_house(b_red,b_blue)
    post = pA*LA/(pA*LA+pB*LB)
    return post


def rand_prob():
    den=rng.choice([2,3,4,5,6,8,10,12])
    num=rng.randint(1, den-1)
    return Fraction(num,den)


def forked_relay_value(route1,route2):
    p1=Fraction(1,1)
    for p in route1: p1*=p
    p2=Fraction(1,1)
    for p in route2: p2*=p
    return 1-(1-p1)*(1-p2)


def random_connected_graph():
    n=rng.randint(4,5)
    nodes=[chr(ord('A')+i) for i in range(n)]
    edges=set()
    # random tree
    for i in range(1,n):
        j=rng.randrange(i)
        edges.add(tuple(sorted((nodes[i], nodes[j]))))
    # add extra edges
    all_edges=[tuple(sorted(e)) for e in itertools.combinations(nodes,2) if tuple(sorted(e)) not in edges]
    rng.shuffle(all_edges)
    for e in all_edges[:rng.randint(0,min(3,len(all_edges)))]:
        edges.add(e)
    adj={u:set() for u in nodes}
    for u,v in edges:
        adj[u].add(v); adj[v].add(u)
    return adj


def gen_silent_census():
    def make_case():
        for _ in range(2000):
            n=rng.randint(6,8)
            full=choose_distinct_ints(n,1,36)
            i,j=sorted(rng.sample(range(n),2))
            known=[v for idx,v in enumerate(full) if idx not in (i,j)]
            m=mean_f(full); v=var_pop(full)
            cands=silent_candidates(known,n,m,v,0,45)
            if len(cands)==1 and cands[0]==tuple(sorted([full[i],full[j]])):
                blanks=[]
                it=iter(known)
                for idx in range(n):
                    if idx in (i,j): blanks.append("_")
                    else: blanks.append(str(next(it)))
                return blanks,m,v,full[i],full[j]
        raise RuntimeError
    ex=[]
    for _ in range(2):
        blanks,m,v,a,b=make_case()
        ex.append(f"Ordered values [{', '.join(blanks)}], average {fmt(m)}, squared-spread seal {fmt(v)} -> {min(a,b)}")
    blanks,m,v,a,b=make_case()
    q=("In the census house, a list of distinct integers was written in increasing order, but two entries were smudged away. "
       "The keeper still preserved the full average and the full squared-spread seal, where the squared-spread seal means the population variance.\n\n"
       "Earlier repaired ledgers:\n" + "\n".join(ex) +
       f"\n\nNow repair this ledger:\nOrdered values [{', '.join(blanks)}], average {fmt(m)}, squared-spread seal {fmt(v)}\n"
       "What is the smaller missing integer?\n\nPlease only give the answer in numerical format and only reply with the answer.")
    return {"template":"silent_census","type":"statistics","question":q,"answer":fmt(min(a,b))}


def gen_moment_swap():
    def make_case():
        for _ in range(2000):
            n=rng.randint(5,8)
            old=[rng.randint(1,25) for _ in range(n)]
            idx=rng.randrange(n)
            a=old[idx]
            b=a+rng.randint(2,12)
            new=old.copy(); new[idx]=b
            m1,v1=mean_f(old), var_pop(old)
            m2,v2=mean_f(new), var_pop(new)
            cands=moment_swap_candidates(n,m1,v1,m2,v2,0,60)
            if len(cands)==1 and cands[0]==(a,b):
                return n,m1,v1,m2,v2,b
        raise RuntimeError
    ex=[]
    for _ in range(2):
        n,m1,v1,m2,v2,b=make_case()
        ex.append(f"Count {n}, old average {fmt(m1)}, old squared-spread seal {fmt(v1)}, new average {fmt(m2)}, new squared-spread seal {fmt(v2)} -> {b}")
    n,m1,v1,m2,v2,b=make_case()
    q=("In the archive of weights, one recorded integer was removed and replaced by a larger corrected integer. "
       "The chamber preserved the old average and squared-spread seal, and also the new average and squared-spread seal after the correction. "
       "Here the squared-spread seal means population variance.\n\n"
       "Earlier correction records:\n" + "\n".join(ex) +
       f"\n\nNow determine the corrected inserted integer for:\nCount {n}, old average {fmt(m1)}, old squared-spread seal {fmt(v1)}, new average {fmt(m2)}, new squared-spread seal {fmt(v2)}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"moment_swap","type":"statistics","question":q,"answer":fmt(b)}


def gen_twin_house():
    def make_case():
        for _ in range(1000):
            pA=Fraction(rng.randint(1,4), rng.randint(5,8))
            a_red,a_blue=rng.randint(2,6),rng.randint(2,6)
            b_red,b_blue=rng.randint(2,6),rng.randint(2,6)
            obs=rng.choice([('R','R'),('R','B'),('B','R'),('B','B')])
            post=posterior_house_A(pA,a_red,a_blue,b_red,b_blue,obs)
            if post!=pA:
                return pA,a_red,a_blue,b_red,b_blue,obs,post
        raise RuntimeError
    obsmap={('R','R'):"red then red",('R','B'):"red then blue",('B','R'):"blue then red",('B','B'):"blue then blue"}
    ex=[]
    for _ in range(2):
        pA,aR,aB,bR,bB,obs,post=make_case()
        ex.append(f"Prior {fmt(pA)}, House A {aR} red {aB} blue, House B {bR} red {bB} blue, observed {obsmap[obs]} -> {fmt(post)}")
    pA,aR,aB,bR,bB,obs,post=make_case()
    q=("In the twin-house registry, a courier first chooses House A or House B using the given prior chance. "
       "From the chosen house, two stones are drawn without replacement in the order shown. The registry then writes the posterior chance that House A had been chosen.\n\n"
       "Earlier reports:\n" + "\n".join(ex) +
       f"\n\nNow determine the posterior seal for:\nPrior {fmt(pA)}, House A {aR} red {aB} blue, House B {bR} red {bB} blue, observed {obsmap[obs]}\n"
       "Please only give the answer in reduced fractional form and only reply with the answer.")
    return {"template":"twin_house","type":"probability","question":q,"answer":fmt(post)}


def gen_forked_relay():
    def make_case():
        for _ in range(1000):
            r1=[rand_prob() for _ in range(rng.randint(2,3))]
            r2=[rand_prob() for _ in range(rng.randint(2,3))]
            ans=forked_relay_value(r1,r2)
            if ans not in (0,1):
                return r1,r2,ans
        raise RuntimeError
    ex=[]
    for _ in range(2):
        r1,r2,ans=make_case()
        ex.append(f"Route one stages {', '.join(fmt(p) for p in r1)}; route two stages {', '.join(fmt(p) for p in r2)} -> {fmt(ans)}")
    r1,r2,ans=make_case()
    q=("In the relay charter, a message may be sent along two independent routes. "
       "A route succeeds only if every one of its stages succeeds, and the message arrives if at least one route succeeds.\n\n"
       "Earlier charters:\n" + "\n".join(ex) +
       f"\n\nNow determine the charter value for:\nRoute one stages {', '.join(fmt(p) for p in r1)}; route two stages {', '.join(fmt(p) for p in r2)}\n"
       "Please only give the answer in reduced fractional form and only reply with the answer.")
    return {"template":"forked_relay","type":"probability","question":q,"answer":fmt(ans)}


def gen_guild_oath():
    def make_case():
        n=rng.randint(5,8); k=rng.randint(2,min(4,n-1))
        return n,k,guild_oath_count(n,k)
    ex=[make_case() for _ in range(2)]
    n,k,ans=make_case()
    q=("In the oath hall, each labeled apprentice must be assigned to exactly one labeled guild. "
       "Every guild must receive at least one apprentice, and the elder guild must receive at least two apprentices.\n\n"
       "Earlier hall records:\n" + "\n".join(f"{a} apprentices, {b} guilds -> {c}" for a,b,c in ex) +
       f"\n\nNow determine the hall's count for:\n{n} apprentices, {k} guilds\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"guild_oath","type":"discrete_math","question":q,"answer":fmt(ans)}


def gen_remainder_street():
    def make_case():
        for _ in range(1000):
            mods=rng.sample([3,4,5,7,8,9,11], rng.choice([2,2,3]))
            conds=[(m,rng.randint(0,m-1)) for m in mods]
            N=rng.randint(25,90)
            ans=count_remainder_street(N, conds)
            if ans>0:
                return N,conds,ans
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    N,conds,ans=make_case()
    def render(N,conds):
        lines=[f"1 <= x <= {N}"]+[f"x leaves remainder {r} when divided by {m}" for m,r in conds]
        return "\n".join(lines)
    q=("In Remainder Street, only house numbers that satisfy every posted remainder rule receive a permit. "
       "The clerk writes how many permitted numbers appear from 1 up to the gate number.\n\n"
       "Earlier permits:\n" + "\n".join(render(Ni,ci).replace("\n","; ") + f" -> {ai}" for Ni,ci,ai in ex) +
       f"\n\nNow determine the permit count for:\n{render(N,conds)}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"remainder_street","type":"number_theory","question":q,"answer":fmt(ans)}


def gen_interval_coprime():
    def make_case():
        m=rng.choice([10,12,14,15,18,20,21,22,24,26,28,30,33,35])
        L=rng.randint(1,40)
        R=L+rng.randint(15,45)
        ans=count_interval_coprime(L,R,m)
        return m,L,R,ans
    ex=[make_case() for _ in range(2)]
    m,L,R,ans=make_case()
    q=("At the coprime toll, a traveler may pass only if the travel number shares no common factor with the gate number. "
       "The tollkeeper records how many numbers in the stated interval may pass.\n\n"
       "Earlier toll records:\n" + "\n".join(f"Gate {m}, interval [{L}, {R}] -> {a}" for m,L,R,a in ex) +
       f"\n\nNow determine the toll count for:\nGate {m}, interval [{L}, {R}]\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"interval_coprime","type":"number_theory","question":q,"answer":fmt(ans)}


def gen_crosswind():
    def make_case():
        for _ in range(5000):
            pts=rng.sample([(x,y) for x in range(-5,6) for y in range(-5,6)],4)
            p1,p2,q1,q2=pts
            den=(p1[0]-p2[0])*(q1[1]-q2[1])-(p1[1]-p2[1])*(q1[0]-q2[0])
            if den==0: continue
            s=line_intersection_sum(p1,p2,q1,q2)
            return p1,p2,q1,q2,s
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    p1,p2,q1,q2,ans=make_case()
    def render_case(c):
        a,b,d,e,s=c
        return f"Path one through {a} and {b}; path two through {d} and {e} -> {fmt(s)}"
    q=("In the crosswind ledger, each pair of sky-paths is traced by two visible markers. "
       "When the paths cross, the clerk records one exact crossing seal: the sum of the crossing point's coordinates.\n\n"
       "Earlier crossings:\n" + "\n".join(render_case(c) for c in ex) +
       f"\n\nNow determine the crossing seal for:\nPath one through {p1} and {p2}; path two through {q1} and {q2}\n"
       "Please only give the answer in reduced fractional form and only reply with the answer.")
    return {"template":"crosswind","type":"geometry","question":q,"answer":fmt(ans)}


def gen_root_mirror():
    def make_case():
        s=rng.randint(-8,10)
        p=rng.randint(-10,12)
        ans=s**3 - 3*p*s
        return s,p,ans
    ex=[make_case() for _ in range(2)]
    s,p,ans=make_case()
    q=("In the root-mirror mint, each monic quadratic hides two roots. "
       "The mint's curve-seal is the sum of the cubes of those roots, and the clerk says it can be found without solving for the roots themselves.\n\n"
       "Earlier mint records:\n" + "\n".join(f"x^2 - ({s})x + ({p}) -> {a}" for s,p,a in ex) +
       f"\n\nNow determine the curve-seal for:\nx^2 - ({s})x + ({p})\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"root_mirror","type":"algebra","question":q,"answer":fmt(ans)}


def gen_layered_box():
    def make_case():
        n=rng.randint(7,16); k=rng.randint(2,5); g=rng.randint(1,3)
        ans=count_layered_box(n,k,g)
        if ans>0:
            return n,k,g,ans
        return make_case()
    ex=[make_case() for _ in range(2)]
    n,k,g,ans=make_case()
    q=("In the layered warehouse, exactly k marked boxes must be placed into a row of n slots. "
       "Between any two marked boxes there must be at least g empty slots. The clerk records how many legal arrangements exist.\n\n"
       "Earlier warehouse records:\n" + "\n".join(f"n={n}, k={k}, g={g} -> {a}" for n,k,g,a in ex) +
       f"\n\nNow determine the warehouse count for:\nn={n}, k={k}, g={g}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"layered_box","type":"combinatorics","question":q,"answer":fmt(ans)}


def gen_merged_variance():
    def make_case():
        for _ in range(5000):
            nA=rng.randint(3,5); nB=rng.randint(3,5)
            A=[rng.randint(1,15) for _ in range(nA)]
            B=[rng.randint(1,15) for _ in range(nB)]
            vB=var_pop(B)
            # prefer integer or simple fraction denominator <=4
            if vB.denominator>4: 
                continue
            mA,vA=mean_f(A),var_pop(A)
            combined=A+B
            M,V=mean_f(combined), var_pop(combined)
            # derive B variance from stats
            # answer exact vB
            return nA,nB,mA,vA,M,V,vB
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    nA,nB,mA,vA,M,V,ans=make_case()
    q=("In the variance archive, two groups of integer measurements were merged. "
       "For Group A, the keeper knows its average and squared-spread seal. For the combined data, he knows the combined average and combined squared-spread seal. "
       "He also knows only the size of Group B, not its squared-spread seal. Here the squared-spread seal means population variance.\n\n"
       "Earlier merge records:\n" + "\n".join(
           f"Group A size {a}, Group B size {b}, Group A average {fmt(c)}, Group A squared-spread {fmt(d)}, combined average {fmt(e)}, combined squared-spread {fmt(f)} -> Group B squared-spread {fmt(g)}"
           for a,b,c,d,e,f,g in ex
       ) +
       f"\n\nNow determine Group B's squared-spread seal for:\nGroup A size {nA}, Group B size {nB}, Group A average {fmt(mA)}, Group A squared-spread {fmt(vA)}, combined average {fmt(M)}, combined squared-spread {fmt(V)}\n"
       "Please only give the answer in reduced numerical form and only reply with the answer.")
    return {"template":"merged_variance","type":"statistics","question":q,"answer":fmt(ans)}


def solve_three_crate(coeffs, vals):
    (a1,b1,c1),(a2,b2,c2),(a3,b3,c3)=coeffs
    v1,v2,v3=vals
    D = a1*(b2*c3-b3*c2)-b1*(a2*c3-a3*c2)+c1*(a2*b3-a3*b2)
    if D==0:
        return None
    Dx = v1*(b2*c3-b3*c2)-b1*(v2*c3-v3*c2)+c1*(v2*b3-v3*b2)
    Dy = a1*(v2*c3-v3*c2)-v1*(a2*c3-a3*c2)+c1*(a2*v3-a3*v2)
    Dz = a1*(b2*v3-b3*v2)-b1*(a2*v3-a3*v2)+v1*(a2*b3-a3*b2)
    return Fraction(Dx,D), Fraction(Dy,D), Fraction(Dz,D)


def gen_three_crate():
    def make_case():
        for _ in range(5000):
            x,y,z=[rng.randint(2,9) for _ in range(3)]
            coeffs=[]
            while len(coeffs)<4:
                tup=(rng.randint(0,4),rng.randint(0,4),rng.randint(0,4))
                if sum(tup)>0 and tup not in coeffs:
                    coeffs.append(tup)
            base=coeffs[:3]; target=coeffs[3]
            vals=[a*x+b*y+c*z for a,b,c in base]
            solved=solve_three_crate(base,vals)
            if solved and all(v.denominator==1 for v in solved):
                ans=target[0]*x+target[1]*y+target[2]*z
                return base,vals,target,ans
        raise RuntimeError
    ex=[]
    for _ in range(2):
        ex.append(make_case())
    base,vals,target,ans=make_case()
    def render(ex):
        base,vals,target,ans=ex
        lines=[f"{a} amber, {b} jade, {c} onyx -> {v}" for (a,b,c),v in zip(base,vals)]
        lines.append(f"{target[0]} amber, {target[1]} jade, {target[2]} onyx -> {ans}")
        return "\n".join(lines)
    q=("In the three-crate ledger, the scribe never wrote the value of a single amber, jade, or onyx stone. "
       "He only recorded bundle values, and the value of a bundle is the sum of its stone values.\n\n"
       "Earlier recovered pages:\n" + "\n\n".join(render(e) for e in ex) +
       f"\n\nNow fill the torn line using these entries:\n" + "\n".join(f"{a} amber, {b} jade, {c} onyx -> {v}" for (a,b,c),v in zip(base,vals)) +
       f"\n{target[0]} amber, {target[1]} jade, {target[2]} onyx -> ?\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"three_crate","type":"algebra","question":q,"answer":fmt(ans)}


def gen_cubic_forge():
    def make_case():
        coeff=[rng.randint(-3,3) for _ in range(4)]
        if coeff[0]==0: coeff[0]=rng.choice([-3,-2,-1,1,2,3])
        xs=choose_distinct_ints(5,-3,5)
        def f(x): return coeff[0]*x**3+coeff[1]*x**2+coeff[2]*x+coeff[3]
        vals=[f(x) for x in xs]
        return xs,vals
    ex1=gen=None
    xs,vals=make_case()
    q=("In the curve forge, each value is produced by one fixed cubic rule.\n\n"
       f"Recorded points from the same cubic:\n" + "\n".join(f"{x} -> {v}" for x,v in zip(xs[:4],vals[:4])) +
       f"\n\nNow determine the value at:\n{xs[4]} -> ?\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"cubic_forge","type":"algebra","question":q,"answer":fmt(vals[4])}


def gen_ballot_prefix():
    def make_case():
        a=rng.randint(4,9); b=rng.randint(1,a-1)
        return a,b,ballot_prefix_count(a,b)
    ex=[make_case() for _ in range(2)]
    a,b,ans=make_case()
    q=("In the ballot chamber, A-votes and B-votes are laid in a line. "
       "The chamber counts only those lines in which, from the first vote onward, A is always strictly ahead of B.\n\n"
       "Earlier chamber records:\n" + "\n".join(f"A votes {a}, B votes {b} -> {c}" for a,b,c in ex) +
       f"\n\nNow determine the chamber count for:\nA votes {a}, B votes {b}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"ballot_prefix","type":"discrete_math","question":q,"answer":fmt(ans)}


def gen_gap_subset_sum():
    def make_case():
        n=rng.randint(8,13); k=rng.randint(2,4)
        # generate by sampling a valid subset
        valid=[]
        for comb in itertools.combinations(range(1,n+1),k):
            if all(comb[i+1]-comb[i]>=2 for i in range(k-1)):
                valid.append(comb)
        subset=rng.choice(valid)
        S=sum(subset)
        ans=count_gap_subset_sum(n,k,S)
        return n,k,S,ans
    ex=[make_case() for _ in range(2)]
    n,k,S,ans=make_case()
    q=("In the archive of separated picks, exactly k numbers are chosen from 1 through n. "
       "No two chosen numbers may be consecutive, and the clerk records how many valid choices have total sum S.\n\n"
       "Earlier records:\n" + "\n".join(f"n={n}, k={k}, S={S} -> {a}" for n,k,S,a in ex) +
       f"\n\nNow determine the archive count for:\nn={n}, k={k}, S={S}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"gap_subset_sum","type":"combinatorics","question":q,"answer":fmt(ans)}


def gen_turn_limited_grid():
    def make_case():
        a=rng.randint(2,6); b=rng.randint(2,6); t=rng.randint(1,a+b-1)
        ans=count_shortest_paths_exact_turns(a,b,t)
        if ans>0:
            return a,b,t,ans
        return make_case()
    ex=[make_case() for _ in range(2)]
    a,b,t,ans=make_case()
    q=("In the grid permit office, a courier moves only east or north along shortest routes from the southwest corner to the northeast corner. "
       "The office records how many shortest routes use exactly t turns.\n\n"
       "Earlier permits:\n" + "\n".join(f"{a} east, {b} north, turns {t} -> {c}" for a,b,t,c in ex) +
       f"\n\nNow determine the permit count for:\n{a} east, {b} north, turns {t}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"turn_limited_grid","type":"discrete_math","question":q,"answer":fmt(ans)}


def gen_divisor_window():
    def make_case():
        N=rng.choice([36,40,42,48,54,56,60,72,84,90,96,108,120,126,140,144,168,180,210,240])
        ds=divisors(N)
        L=rng.randint(1,max(2,N//3))
        R=rng.randint(L,min(N, L+rng.randint(10,60)))
        ans=sum(1 for d in ds if L<=d<=R)
        return N,L,R,ans
    ex=[make_case() for _ in range(2)]
    N,L,R,ans=make_case()
    q=("In the divisor window census, the keeper looks only at divisors of N that lie inside the stated window [L, R], and he records how many divisors appear there.\n\n"
       "Earlier windows:\n" + "\n".join(f"N={N}, window [{L}, {R}] -> {a}" for N,L,R,a in ex) +
       f"\n\nNow determine the census value for:\nN={N}, window [{L}, {R}]\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"divisor_window","type":"number_theory","question":q,"answer":fmt(ans)}


def gen_quadratic_residue():
    def make_case():
        for _ in range(1000):
            m=rng.choice([5,7,8,9,11,12,13,15,16,17,18,20])
            a=rng.randint(0,m-1)
            N=rng.randint(20,80)
            ans=count_quadratic_residue(N,a,m)
            if ans>0:
                return N,a,m,ans
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    N,a,m,ans=make_case()
    q=("In the square-remainder registry, a number x is accepted only if its square leaves the stated remainder when divided by the stated modulus. "
       "The registry records how many accepted numbers lie from 1 up to N.\n\n"
       "Earlier registry lines:\n" + "\n".join(f"1 <= x <= {N}, x^2 leaves remainder {a} mod {m} -> {c}" for N,a,m,c in ex) +
       f"\n\nNow determine the registry count for:\n1 <= x <= {N}, x^2 leaves remainder {a} mod {m}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"quadratic_residue","type":"number_theory","question":q,"answer":fmt(ans)}


def gen_exact_gcd():
    def make_case():
        for _ in range(1000):
            M=rng.choice([12,18,20,24,30,36,40,45,48,54,60,72,84,90])
            d=rng.choice(divisors(M))
            N=rng.randint(20,90)
            ans=count_exact_gcd(N,M,d)
            if ans>0:
                return N,M,d,ans
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    N,M,d,ans=make_case()
    q=("At the exact-gcd toll, a travel number passes only if its greatest common divisor with the gate number M is exactly d. "
       "The tollkeeper records how many passing numbers lie from 1 to N.\n\n"
       "Earlier toll records:\n" + "\n".join(f"N={N}, M={M}, d={d} -> {c}" for N,M,d,c in ex) +
       f"\n\nNow determine the toll count for:\nN={N}, M={M}, d={d}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"exact_gcd","type":"number_theory","question":q,"answer":fmt(ans)}


def gen_conditional_reward():
    def make_case():
        r=rng.randint(2,6); b=rng.randint(2,6); d=rng.choice([2,3])
        if d>r+b: d=2
        Rv=rng.randint(4,12); Bv=rng.randint(1,5)
        ans=expected_conditional_reward(r,b,Rv,Bv,d)
        return r,b,Rv,Bv,d,ans
    ex=[make_case() for _ in range(2)]
    r,b,Rv,Bv,d,ans=make_case()
    q=("In the reward bag room, a bag contains red stones worth R points each and blue stones worth B points each. "
       "Exactly d stones are drawn without replacement. The registry records the expected total reward, given that at least one red stone was drawn.\n\n"
       "Earlier registry lines:\n" + "\n".join(f"{r} red, {b} blue, R={Rv}, B={Bv}, d={d} -> {fmt(a)}" for r,b,Rv,Bv,d,a in ex) +
       f"\n\nNow determine the registry value for:\n{r} red, {b} blue, R={Rv}, B={Bv}, d={d}\n"
       "Please only give the answer in reduced fractional form and only reply with the answer.")
    return {"template":"conditional_reward","type":"probability","question":q,"answer":fmt(ans)}


def gen_graph_walk():
    def make_case():
        for _ in range(1000):
            adj=random_connected_graph()
            nodes=sorted(adj)
            s,t=rng.sample(nodes,2)
            L=rng.randint(2,5)
            ans=count_walks(adj,s,t,L)
            if ans>0:
                return adj,s,t,L,ans
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    adj,s,t,L,ans=make_case()
    def render(case):
        adj,s,t,L,ans=case
        edges=sorted({tuple(sorted((u,v))) for u in adj for v in adj[u] if u<v})
        return f"Edges {edges}, start {s}, end {t}, length {L} -> {ans}"
    q=("In the walk registry, a patrol may traverse one edge per step and may revisit vertices and edges. "
       "The registry records how many walks of exactly the stated length go from the start vertex to the end vertex.\n\n"
       "Earlier registry lines:\n" + "\n".join(render(c) for c in ex) +
       f"\n\nNow determine the registry value for:\n{render((adj,s,t,L,'?')).replace('-> ?','')}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"graph_walk","type":"discrete_math","question":q,"answer":fmt(ans)}


def gen_deranged_seating():
    def make_case():
        for _ in range(1000):
            n=rng.randint(4,8); k=rng.randint(0,n)
            ans=math.comb(n,k)*derangement(n-k)
            if ans>0:
                return n,k,ans
        raise RuntimeError
    ex=[make_case() for _ in range(2)]
    n,k,ans=make_case()
    q=("In the seating chamber, n labeled guests choose seats among n labeled seats. "
       "The chamber records how many seating arrangements leave exactly k guests in their own original seats.\n\n"
       "Earlier chamber records:\n" + "\n".join(f"n={n}, exactly k={k} fixed seats -> {a}" for n,k,a in ex) +
       f"\n\nNow determine the chamber count for:\nn={n}, exactly k={k} fixed seats\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"deranged_seating","type":"combinatorics","question":q,"answer":fmt(ans)}


def gen_block_partition():
    def make_case():
        n=rng.randint(4,9); k=rng.randint(1,n//2)
        ans=block_partitions_min2(n,k)
        return n,k,ans
    ex=[make_case() for _ in range(2)]
    n,k,ans=make_case()
    q=("In the partition hall, n labeled tokens are to be split into k unlabeled groups, and every group must contain at least two tokens. "
       "The hall records how many such partitions exist.\n\n"
       "Earlier hall records:\n" + "\n".join(f"n={n}, k={k} -> {a}" for n,k,a in ex) +
       f"\n\nNow determine the hall's count for:\nn={n}, k={k}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"block_partition","type":"discrete_math","question":q,"answer":fmt(ans)}


def gen_maximum_draw():
    def make_case():
        vals=[rng.randint(1,12) for _ in range(rng.randint(4,7))]
        ans=expected_max_of_two(vals)
        return vals,ans
    ex=[make_case() for _ in range(2)]
    vals,ans=make_case()
    q=("In the maximum-draw room, two stones are drawn uniformly without replacement from the listed values, and the house value is the expected larger of the two drawn values.\n\n"
       "Earlier room records:\n" + "\n".join(f"{vals} -> {fmt(a)}" for vals,a in ex) +
       f"\n\nNow determine the house value for:\n{vals}\n"
       "Please only give the answer in reduced fractional form and only reply with the answer.")
    return {"template":"maximum_draw","type":"probability","question":q,"answer":fmt(ans)}


def gen_triangle_area16():
    def make_case():
        # choose Heronian triple from set
        triples=[(3,4,5),(5,5,6),(5,12,13),(6,8,10),(7,15,20),(8,15,17),(9,10,17),(10,13,13),(13,14,15)]
        a,b,c=rng.choice(triples)
        ans=triangle_area16(a,b,c)
        return a,b,c,ans
    ex=[make_case() for _ in range(2)]
    a,b,c,ans=make_case()
    q=("In the triangle ledger, the clerk does not record the area itself. "
       "For a triangle with side lengths a, b, c, he records sixteen times the square of the area.\n\n"
       "Earlier ledger lines:\n" + "\n".join(f"a={a}, b={b}, c={c} -> {ans}" for a,b,c,ans in ex) +
       f"\n\nNow determine the ledger value for:\na={a}, b={b}, c={c}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"triangle_area16","type":"geometry","question":q,"answer":fmt(ans)}


def gen_polygon_double_area():
    def make_case():
        # simple convex-ish polygon from shuffled points around center
        pts=rng.sample([(x,y) for x in range(-3,6) for y in range(-3,6)], rng.randint(4,6))
        cx=sum(x for x,y in pts)/len(pts); cy=sum(y for x,y in pts)/len(pts)
        pts=sorted(pts, key=lambda p: math.atan2(p[1]-cy,p[0]-cx))
        # ensure no duplicates and nonzero area
        da=polygon_double_area(pts)
        if da>0:
            return pts,da
        return make_case()
    ex=[make_case() for _ in range(2)]
    pts,ans=make_case()
    q=("In the polygon register, the clerk traces the listed lattice vertices in order and records twice the enclosed area.\n\n"
       "Earlier register lines:\n" + "\n".join(f"{pts} -> {a}" for pts,a in ex) +
       f"\n\nNow determine the register value for:\n{pts}\n"
       "Please only give the answer in numerical format and only reply with the answer.")
    return {"template":"polygon_double_area","type":"geometry","question":q,"answer":fmt(ans)}


def generate_dataset(counts):
    items=[]
    seen=set()
    tid=1
    template_attempts={}
    for name,quota in counts.items():
        g=gens_map[name]
        got=0; attempts=0
        while got<quota and attempts<quota*500:
            attempts+=1
            item=g()
            q=item["question"]
            if q in seen:
                continue
            seen.add(q)
            item["id"]=tid
            tid+=1
            items.append(item)
            got+=1
        template_attempts[name]=attempts
        if got<quota:
            raise RuntimeError(f"{name} only {got}/{quota}")
    return items, template_attempts


gens_map = {
    'silent_census': gen_silent_census,
    'moment_swap': gen_moment_swap,
    'twin_house': gen_twin_house,
    'forked_relay': gen_forked_relay,
    'guild_oath': gen_guild_oath,
    'remainder_street': gen_remainder_street,
    'interval_coprime': gen_interval_coprime,
    'crosswind': gen_crosswind,
    'root_mirror': gen_root_mirror,
    'layered_box': gen_layered_box,
    'merged_variance': gen_merged_variance,
    'three_crate': gen_three_crate,
    'cubic_forge': gen_cubic_forge,
    'ballot_prefix': gen_ballot_prefix,
    'gap_subset_sum': gen_gap_subset_sum,
    'turn_limited_grid': gen_turn_limited_grid,
    'divisor_window': gen_divisor_window,
    'quadratic_residue': gen_quadratic_residue,
    'exact_gcd': gen_exact_gcd,
    'conditional_reward': gen_conditional_reward,
    'graph_walk': gen_graph_walk,
    'deranged_seating': gen_deranged_seating,
    'block_partition': gen_block_partition,
    'maximum_draw': gen_maximum_draw,
    'triangle_area16': gen_triangle_area16,
    'polygon_double_area': gen_polygon_double_area,
}

COUNTS = {'silent_census': 55, 'moment_swap': 60, 'twin_house': 60, 'forked_relay': 45, 'guild_oath': 50, 'remainder_street': 60, 'interval_coprime': 45, 'crosswind': 40, 'root_mirror': 35, 'layered_box': 45, 'merged_variance': 60, 'three_crate': 40, 'cubic_forge': 40, 'ballot_prefix': 45, 'gap_subset_sum': 60, 'turn_limited_grid': 60, 'divisor_window': 35, 'quadratic_residue': 50, 'exact_gcd': 50, 'conditional_reward': 60, 'graph_walk': 45, 'deranged_seating': 35, 'block_partition': 60, 'maximum_draw': 40, 'triangle_area16': 35, 'polygon_double_area': 35}


if __name__ == "__main__":
    items, _ = generate_dataset(COUNTS)
    public_rows = [{"id": item["id"], "type": item["type"], "question": item["question"], "answer": item["answer"]} for item in items]
    internal_rows = [{"id": item["id"], "template": item["template"], "type": item["type"], "question": item["question"], "answer": item["answer"]} for item in items]
    pd.DataFrame(public_rows).to_csv("mathrulebench_hard_1245.csv", index=False)
    pd.DataFrame(internal_rows).to_csv("mathrulebench_hard_1245_with_templates.csv", index=False)
    pd.DataFrame([{"template": k, "count": v} for k, v in COUNTS.items()]).to_csv("mathrulebench_hard_1245_template_counts.csv", index=False)
    with open("mathrulebench_hard_1245.jsonl", "w", encoding="utf-8") as f:
        for row in public_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generated {len(items)} items.")
