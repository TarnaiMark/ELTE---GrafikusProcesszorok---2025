#include <vector>

template<typename T>
void cpu_matmul_naive(std::vector<T>& C, std::vector<T> const& A, std::vector<T> const& B, int N) 
{
    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            T sum = 0;
            for(int k=0; k<N; ++k)
            {
                sum += A[y*N+k] * B[k*N+x];
            }
            C[y*N+x] = sum;
        }
    }
}

template<typename T, int MBS = 8>
void cpu_matmul_improved(std::vector<T>& C, std::vector<T> const& A, std::vector<T> const& B, int N) 
{
    for( int by=0; by<N/MBS; ++by ) //block index 1
    {
        for( int bx=0; bx<N/MBS; ++bx ) //block index 2
        {
            for( int bk=0; bk<N/MBS; ++bk ) //block index 3
            {
                auto y0 = by * MBS;
                auto x0 = bx * MBS;
                auto k0 = bk * MBS;
                for( int y=0; y<MBS; ++y )
                {
                    auto yy = y0 + y;
                    for( int x=0; x<MBS; ++x )
                    {
                        auto xx = x0 + x;
                        T sum = 0;
                        for( int k=0; k<MBS; ++k )
                        {
                            sum += A[yy*N+k0+k] * B[(k0+k)*N+xx];
                        }
                        C[yy*N+xx] += sum;
                    }
                }
            }
        }
    }
}