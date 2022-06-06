! routine to find one interpolated value given a table of n values
! need a routine that handles a vector of inputs. But this is what
! romberg requires.
SUBROUTINE polint(xx,yy,n,x,y,err,c,d)
  !
  !
  INTEGER :: n
  REAL*8, dimension(n) :: xx,yy
  REAL*8, dimension(n) :: c(n),d(n)
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
END SUBROUTINE polint
