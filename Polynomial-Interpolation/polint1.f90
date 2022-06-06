! routine to find one interpolated value given a table of n values
! need a routine that handles a vector of inputs. But this is what
! romberg requires.
SUBROUTINE polint1(xx,yy,n,x,y,err,c,d)
  !
  !
  INTEGER :: n
  REAL*8, dimension(n) :: xx,yy
  REAL*8, dimension(n,n) ,intent(out) :: c(n,n),d(n,n)
  !f2py intent(hide) c,d
  REAL*8, intent(out) :: err
  REAL*8 :: x
  REAL*8, intent(out) :: y
  INTEGER :: i,m,ns
  REAL*8 :: den,dif,dift,ho,hp,w
  ns=1
  dif=ABS(x-xx(1))
  ! very inefficient algorithm to locate the n points in table
  DO i=1,n
     dift=ABS(x-xx(i))
     IF (dift.LT.dif) THEN
        ns=i
        dif=dift
     ENDIF
     c(1,i)=yy(i)
     d(1,i)=yy(i)
  END DO
  y=yy(ns)
  ns=ns-1
  DO m=1,n-1
     DO i=1,n-m
        ho=xx(i)-x
        hp=xx(i+m)-x
        w=c(m,i+1)-d(m,i)
        den=ho-hp
        IF(den.EQ.0.)PAUSE 'failure in polint'
        den=w/den
        d(m+1,i)=hp*den
        c(m+1,i)=ho*den
     END DO
     IF (2*ns.LT.n-m)THEN
        err=c(m,ns+1)
     ELSE
        err=d(m,ns)
        ns=ns-1
     ENDIF
     y=y+err
  END DO
  RETURN
END SUBROUTINE polint1
