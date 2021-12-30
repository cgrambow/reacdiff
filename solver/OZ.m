function [h,c,hk,ck] = OZ(u,closure,tol)
  %solve for C2 and h using Ornstein-Zernike and closure
  xi = zeros(size(u));
  while true
    switch closure
    case 'HNC'
      h = exp(-u+xi)-1;
    case 'PY'
      h = exp(-u) .* (xi+1)-1;
    end
    hk = psf2otf(h);
    hk = -0.8/min(hk(:))*hk;
%     hk = fftn(h);
    ck = hk./(hk+1);
    c = otf2psf(ck);
%     c = ifftn(ck,'symmetric');
    xinew = h - c;
    if norm(xinew(:)-xi(:)) < tol
      break;
    end
    xi = xinew;
  end
