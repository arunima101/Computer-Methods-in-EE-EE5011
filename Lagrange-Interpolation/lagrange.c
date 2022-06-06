
#include<stdio.h>
#include<math.h>
#include<stdlib.h>


double lintp(double *xx,double *yy, double x, int n)
{
    int i,j;
    double p,val;
    
    for (i=0;i<n;i++)
    {
        p=1;
         for(j=0;j<n;j++)
         {
             if(i!=j)
             {
                p=p*(x-xx[j])/(xx[i]-xx[j]);
             }
         }
     
        val = val + p*yy[i];
    }
  return val;
}

int locate(double *xx,int t,double x)
{
    unsigned long ju,jm,jl;
    int ascnd;
    int j;
    jl=0;
    ju=t;
    ascnd=(xx[t-1]>=xx[0]);
    while (ju-jl > 1){
        jm= (ju+jl) >>1;
        if(x>=xx[jm] == ascnd)
            jl=jm;
        else
            ju=jm;
    }
    if (x==xx[0]) j=0;
    else if(x==xx[t-1]) j=t-1;
    else j=jl;
    return j ;
}    


void main()
{
    int i,j=0;
    double xx[9],yy[9],valuex[4],valuey[4];
    xx[0]=0;
    yy[0]=0;
    for(i=1;i<9;i++)
    {
         xx[i]=xx[i-1]+ M_PI_4;
         yy[i]=sin(xx[i]);
    }
    //For 8th order interpolation
    double var=0;
    FILE *fptr;
    fptr = fopen("output.txt","w+");
    for(i=0;i<100;i++)
    {
        fprintf(fptr,"%lf\t",var);
    	fprintf(fptr,"%lf\n",lintp(xx,yy,var,9));
        var=var+ (M_PI_2/25);
        
    }
    fclose(fptr);
    int v; 
    
    //For cubic interpolation
    var=0;
    FILE *fptr1;
    fptr1 = fopen("output1.txt","w+");
    for (int i=0;i<100;i++){
    	fprintf(fptr1,"%lf\t",var);
     v = locate(xx,9,var);
     if(v<=2){ 
    	for (int c=0;c<4;c++){
    	 	valuex[c]=xx[c];
    	 	valuey[c]=yy[c];
        }
     }
     
     if(v>=3){
         if(v==7 || v==8){
             for(j=0;j<4;j++)
             {
                 valuex[j]=xx[j+5];
                 valuey[j]=yy[j+5];
             }
         }
         j=0; 
    	for (int d=v-2;d<=v+1;d++){
    	 	valuex[j]=xx[d];
    	 	valuey[j]=yy[d];
            j++;
        }
     }

     
    	fprintf(fptr1,"%lf\n",lintp(valuex,valuey,var,4));
        var=var+ (M_PI_2/25);
    }   
    fclose(fptr1); 

    
    
}