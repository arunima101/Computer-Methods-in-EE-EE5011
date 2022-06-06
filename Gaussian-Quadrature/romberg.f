* routine to find one interpolated value given a table of n values
* need a routine that handles a vector of inputs. But this is what
* romberg requires.
      SUBROUTINE polint(xx,yy,n,x,y,err,c,d)
c
cf2py intent(out) :: y
cf2py intent(out) :: err
cf2py intent(hide) :: n
cf2py intent(hide) :: c
cf2py intent(hide) :: d
cf2py integer :: n
cf2py real*8,dimension(n) :: xx
cf2py real*8,dimension(n) :: yy
cf2py real*8,dimension(n) :: c
cf2py real*8,dimension(n) :: d
cf2py real*8 :: x
cf2py real*8 :: y
cf2py real*8 :: err
c
      INTEGER n
      REAL*8 err,x,y,xx(n),yy(n)
      INTEGER i,m,ns
      REAL*8 den,dif,dift,ho,hp,w,c(n),d(n)
      ns=1
      dif=ABS(x-xx(1))
      DO i=1,n
        dift=ABS(x-xx(i))
        IF (dift.LT.dif) THEN
          ns=i
          dif=dift
        ENDIF
        c(i)=yy(i)
        d(i)=yy(i)
      END DO
      y=yy(ns)
      ns=ns-1
      DO m=1,n-1
        DO i=1,n-m
           ho=xx(i)-x
           hp=xx(i+m)-x
           w=c(i+1)-d(i)
           den=ho-hp
           IF(den.EQ.0.)PAUSE 'failure in polint'
           den=w/den
           d(i)=hp*den
           c(i)=ho*den
        END DO
        IF (2*ns.LT.n-m)THEN
           err=c(ns+1)
        ELSE
           err=d(ns)
           ns=ns-1
        ENDIF
        y=y+err
      END DO
      RETURN
      END

      SUBROUTINE trapzd(func,a,b,sin,sout,n)
c
cf2py intent(out) :: sout
cf2py integer :: n
cf2py real*8 :: a
cf2py real*8 :: b
cf2py real*8 :: sin
cf2py real*8 :: sout
      EXTERNAL func
cf2py a = func(b)
c
      INTEGER n
      REAL*8 a,b,sin,sout,func
      INTEGER it,j
      REAL*8 del,sum,tnm,x
      IF (n.EQ.1) THEN
        sout=0.5d0*(b-a)*(func(a)+func(b))
      ELSE
        it=2**(n-2)
        tnm=it
        del=(b-a)/tnm
        x=a+0.5d0*del
        sum=0.d0
        DO j=1,it
           sum=sum+func(x)
           x=x+del
        END DO
        sout=0.5d0*(sin+(b-a)*sum/tnm)
      ENDIF
      RETURN
      END


      SUBROUTINE qromb(func,a,b,ss,dss,numcalls,EPS,K,c,d)
c
cf2py intent(out) :: ss
cf2py intent(out) :: dss
cf2py intent(out) :: numcalls
cf2py real*8,optional :: EPS /1.e-6/
cf2py integer,optional :: K /5/
cf2py intent(hide) :: c
cf2py intent(hide) :: d
cf2py real*8,dimension(K) :: c
cf2py real*8,dimension(K) :: d
cf2py real*8 :: a
cf2py real*8 :: b
cf2py real*8 :: ss
      EXTERNAL func
cf2py real*8 :: y1
cf2py real*8 :: y2
cf2py y1 = func(y2)
c
      INTEGER JMAX,JMAXP,K,numcalls
      REAL*8 a,b,func,ss,EPS,c(k),d(k)
c     PARAMETER (EPS=1.e-6, JMAX=20, JMAXP=JMAX+1, K=5, KM=K-1)
      PARAMETER (JMAX=20,JMAXP=JMAX+1)
      INTEGER KM
      INTEGER j
      REAL*8 dss,h(JMAXP),s(JMAXP),sold
      h(1)=1.d0
      KM=K-1
      sold=0.0d0
      DO j=1,JMAX
        CALL trapzd(func,a,b,sold,s(j),j)
        IF (j.GE.K) THEN
          CALL polint(h(j-KM),s(j-KM),K,0.d0,ss,dss,c,d)
          IF (ABS(dss).LE.EPS*ABS(ss))THEN
             numcalls=2**(j-1)+1
             RETURN
          END IF
        ENDIF
        sold=s(j)
        h(j+1)=0.25d0*h(j)
      END DO
c     no convergence. Return this via numcalls
      numcalls=-1
      END

