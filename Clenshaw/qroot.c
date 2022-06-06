#include<stdio.h>
#include<math.h>
#include<complex.h>

int sgn(float x)
{
    if(x>0)
     return 1;
    if(x<0)
     return -1;
    else
     return 0;

}
void main()
{   
    int N=10000,j=0;
    float complex z1[N],z2[N],z3[N],z4[N];
    double complex e1[N],e2[N];
    double alpha[N],i;
    FILE *fp1,*fp2;
    fp1=fopen("error_from_exact_formula.txt","w");
    fp2=fopen("error_from_accurate_formula.txt","w");
    for (i=0;i<=100;i=i+0.01)
    {
       alpha[j]=i;
       j++;
    } 
 
    for (j=0;j<N;j++)
    {
      double complex y=0;
      if(fabs(alpha[j])<1)
         y=I*sqrt(1-pow(alpha[j],2));
      else
        y= sqrt(pow(alpha[j],2)-1);
      
       e1[j]=-(alpha[j])+ y;
       e2[j]=-(alpha[j])- y;
       
    }
      
   
    for (j=0;j<N;j++)
    {
      float complex y=0;
      if(fabs(alpha[j])<1)
         y=I*sqrt(1-pow(alpha[j],2));
      else
        y= sqrt(pow(alpha[j],2)-1);

       z1[j]=-(alpha[j])+ y;
       z2[j]=-(alpha[j])- y;
       fprintf(fp1,"%.15f\t",alpha[j]);
       fprintf(fp1,"%.15f\t",cabs(z1[j]-e1[j]));
       fprintf(fp1,"%.15f\n",cabs(z2[j]-e2[j]));
       
    }

    for (j=0;j<N;j++)
    {
       float complex p = 0;
       if(fabs(alpha[j])<1)
         p=I*sqrt(1-pow(alpha[j],2));
       else
        p= sqrt(pow(alpha[j],2)-1);
        z3[j]=1/(-(alpha[j])-p);
        z4[j]=1/(-(alpha[j])+p);
       
       fprintf(fp2,"%.15f\t",alpha[j]);
       fprintf(fp2,"%.15lf\t",cabs(z3[j]-e1[j]));
       fprintf(fp2,"%.15lf\n",cabs(z4[j]-e2[j]));
       

    }
    
    fclose(fp1);
    fclose(fp2);


}
