const navToggle = document.querySelector('.nav-toggle');
const nav = document.querySelector('.nav');
const revealEls = document.querySelectorAll('.reveal');
const figureCards = document.querySelectorAll('.figure-card');
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const lightboxTitle = document.getElementById('lightbox-title');
const lightboxClose = document.querySelector('.lightbox-close');
const lightboxBackdrop = document.querySelector('.lightbox-backdrop');

function openNav() {
  nav.classList.add('is-open');
  navToggle.setAttribute('aria-expanded', 'true');
}

function closeNav() {
  nav.classList.remove('is-open');
  navToggle.setAttribute('aria-expanded', 'false');
}

navToggle?.addEventListener('click', () => {
  if (nav.classList.contains('is-open')) {
    closeNav();
  } else {
    openNav();
  }
});

nav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    if (window.innerWidth <= 860) {
      closeNav();
    }
  });
});

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.14 });

revealEls.forEach((el) => revealObserver.observe(el));

function openLightbox(src, alt, title) {
  lightboxImage.src = src;
  lightboxImage.alt = alt;
  lightboxTitle.textContent = title || alt;
  lightbox.classList.add('is-open');
  lightbox.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  lightboxClose.focus();
}

function closeLightbox() {
  lightbox.classList.remove('is-open');
  lightbox.setAttribute('aria-hidden', 'true');
  lightboxImage.src = '';
  lightboxImage.alt = '';
  document.body.style.overflow = '';
}

figureCards.forEach((card) => {
  card.addEventListener('click', () => {
    openLightbox(card.dataset.src, card.dataset.alt, card.dataset.title);
  });
});

lightboxClose?.addEventListener('click', closeLightbox);
lightboxBackdrop?.addEventListener('click', closeLightbox);

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    if (lightbox.classList.contains('is-open')) {
      closeLightbox();
    }
    if (nav.classList.contains('is-open')) {
      closeNav();
    }
  }
});

const counters = document.querySelectorAll('[data-count]');
const countObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    const el = entry.target;
    const target = el.dataset.count;
    if (!target || el.dataset.animated === 'true') return;
    el.dataset.animated = 'true';

    if (target.includes('.')) {
      const value = Number.parseFloat(target);
      animateNumber(el, value, 1100, (num) => {
        el.textContent = num.toFixed(target.split('.')[1].length);
      });
    } else {
      const value = Number.parseInt(target, 10);
      animateNumber(el, value, 1100, (num) => {
        el.textContent = Number.isInteger(value) ? Math.round(num).toLocaleString() : num.toString();
      });
    }
    countObserver.unobserve(el);
  });
}, { threshold: 0.45 });

function animateNumber(el, target, duration, render) {
  const start = performance.now();
  const initial = 0;
  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = initial + (target - initial) * eased;
    render(value);
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

counters.forEach((counter) => countObserver.observe(counter));
