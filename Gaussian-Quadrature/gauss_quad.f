* function used by gauss-legendre. Private function
      REAL*8 FUNCTION gammln(xx)
      REAL*8 xx
      INTEGER j
      REAL*8 ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     *24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     *-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      DO j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
      END DO
      gammln=tmp+LOG(stp*ser/x)
      RETURN
      END
* gauss legendre quadrature that solves the problem
*   \int_{x1}^{x2} f(x)dx = \sum_{j=1}^n w_j f(x_j)
* This routine accepts x1,x2 and n and returns the vectors x and w.
* the function is not required to be passed to it.
      SUBROUTINE gauleg(x1,x2,x,w,n)
c
cf2py intent(out) :: x
cf2py intent(out) :: w
cf2py integer :: n
cf2py real*8 :: x1
cf2py real*8 :: x2
cf2py real*8 :: x
cf2py real*8 :: w
c
      INTEGER n
      REAL*8 x1,x2,x(n),w(n)
      REAL*8 EPS
      PARAMETER (EPS=3.d-14)
      INTEGER i,j,m
      REAL*8 p1,p2,p3,pp,xl,xm,z,z1
      m=(n+1)/2
      xm=0.5d0*(x2+x1)
      xl=0.5d0*(x2-x1)
      DO i=1,m
         z=COS(3.141592654d0*(i-.25d0)/(n+.5d0))
         DO WHILE(.TRUE.)
            p1=1.d0
            p2=0.d0
            DO j=1,n
               p3=p2
               p2=p1
               p1=((2.d0*j-1.d0)*z*p2-(j-1.d0)*p3)/j
            END DO
            pp=n*(z*p1-p2)/(z*z-1.d0)
            z1=z
            z=z1-p1/pp
            IF(ABS(z-z1).LE.EPS)EXIT
         END DO
         x(i)=xm-xl*z
         x(n+1-i)=xm+xl*z
         w(i)=2.d0*xl/((1.d0-z*z)*pp*pp)
         w(n+1-i)=w(i)
      END DO
      RETURN
      END
* gauss-laguare quadrature that solves the problem
      SUBROUTINE gaulag(x,w,n,alf)
c
cf2py intent(out) :: x
cf2py intent(out) :: w
cf2py integer :: n
cf2py real*8 :: alf
cf2py real*8 :: x
cf2py real*8 :: w
c
      INTEGER n,MAXIT
      REAL*8 alf,w(n),x(n)
      REAL*8 EPS
      PARAMETER (EPS=3.D-14,MAXIT=10)
C     USES gammln
      INTEGER i,its,j
      REAL*8 ai,gammln
      REAL*8 p1,p2,p3,pp,z,z1
      DO i=1,n
         IF(i.EQ.1)THEN
            z=(1.d0+alf)*(3.d0+.92d0*alf)/(1.d0+2.4d0*n+1.8d0*alf)
         ELSE IF(i.EQ.2)THEN
            z=z+(15.d0+6.25d0*alf)/(1.d0+.9d0*alf+2.5d0*n)
         ELSE
            ai=i-2
            z=z+((1.d0+2.55d0*ai)/(1.9d0*ai)+1.26d0*ai*alf/
     *           (1.+3.5d0*ai))*(z-x(i-2))/(1.d0+.3d0*alf)
         ENDIF
         DO its=1,MAXIT
            p1=1.d0
            p2=0.d0
            DO j=1,n
               p3=p2
               p2=p1
               p1=((2*j-1+alf-z)*p2-(j-1+alf)*p3)/j
            END DO
            pp=(n*p1-(n+alf)*p2)/z
            z1=z
            z=z1-p1/pp
            IF(ABS(z-z1).LE.EPS)THEN
               EXIT
            ENDIF
         END DO
         x(i)=z
         w(i)=-exp(gammln(alf+n)-gammln(dble(n)))/(pp*n*p2)
      END DO
      RETURN
      END

      SUBROUTINE gauher(x,w,n)
c
cf2py intent(out) :: x
cf2py intent(out) :: w
cf2py integer :: n
cf2py real*8 :: x
cf2py real*8 :: w
c
      INTEGER n,MAXIT
      REAL*8 w(n),x(n)
      REAL*8 EPS,PIM4
      PARAMETER (EPS=3.D-14,PIM4=.7511255444649425D0,MAXIT=10)
      INTEGER i,its,j,m
      REAL*8 p1,p2,p3,pp,z,z1
      m=(n+1)/2
      DO i=1,m
         IF(i.EQ.1)THEN
            z=SQRT(DBLE(2*n+1))-1.85575d0*(2*n+1)**(-.16667d0)
         ELSE IF(i.EQ.2)THEN
            z=z-1.14d0*n**.426d0/z
         ELSE IF (i.EQ.3)THEN
            z=1.86d0*z-.86d0*x(1)
         ELSE IF (i.EQ.4)THEN
            z=1.91d0*z-.91d0*x(2)
         ELSE
            z=2.d0*z-x(i-2)
         ENDIF
         DO its=1,MAXIT
            p1=PIM4
            p2=0.d0
            DO j=1,n
               p3=p2
               p2=p1
               p1=z*sqrt(2.d0/j)*p2-sqrt(dble(j-1)/dble(j))*p3
            END DO
            pp=sqrt(2.d0*n)*p2
            z1=z
            z=z1-p1/pp
            IF(ABS(z-z1).LE.EPS)EXIT
         END DO
         x(i)=z
         x(n+1-i)=-z
         w(i)=2.d0/(pp*pp)
         w(n+1-i)=w(i)
      END DO
      RETURN
      END
