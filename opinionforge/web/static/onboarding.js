/**
 * OpinionForge Onboarding Wizard
 *
 * Vanilla JS logic for the 5-step setup flow: step navigation,
 * provider selection, connection testing, tour carousel, and
 * test generation.
 *
 * No external dependencies — pure DOM manipulation.
 */

var setupWizard = (function () {
    'use strict';

    /* ------------------------------------------------------------------ */
    /* State                                                               */
    /* ------------------------------------------------------------------ */

    var currentStep = 1;
    var maxVisitedStep = 1;
    var selectedProvider = null;
    var selectedSearch = null;
    var connectionTested = {};  /* { providerType: true } */
    var tourSlide = 0;
    var totalTourSlides = 4;

    /* ------------------------------------------------------------------ */
    /* Step navigation                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Navigate to a specific step if it has been unlocked.
     * @param {number} step - Step number (1-5).
     */
    function goToStep(step) {
        if (step < 1 || step > 5) return;
        if (step > maxVisitedStep + 1) return;  /* Can only advance one ahead */

        /* Hide current step */
        var currentEl = document.getElementById('step-' + currentStep);
        if (currentEl) currentEl.classList.remove('active');

        /* Show target step */
        var targetEl = document.getElementById('step-' + step);
        if (targetEl) targetEl.classList.add('active');

        /* Update progress indicators */
        for (var i = 1; i <= 5; i++) {
            var progressEl = document.getElementById('progress-step-' + i);
            if (!progressEl) continue;
            progressEl.classList.remove('active', 'completed');
            if (i < step) {
                progressEl.classList.add('completed');
                progressEl.disabled = false;
            } else if (i === step) {
                progressEl.classList.add('active');
                progressEl.disabled = false;
            } else if (i <= maxVisitedStep) {
                progressEl.disabled = false;
            } else {
                progressEl.disabled = true;
            }
        }

        currentStep = step;
        if (step > maxVisitedStep) maxVisitedStep = step;
    }

    /** Advance to the next step. */
    function next() {
        goToStep(currentStep + 1);
    }

    /** Go back to the previous step. */
    function back() {
        goToStep(currentStep - 1);
    }

    /* ------------------------------------------------------------------ */
    /* Provider selection (Step 2)                                         */
    /* ------------------------------------------------------------------ */

    /**
     * Select an LLM provider card.
     * @param {string} providerType - One of: ollama, anthropic, openai, openai_compatible
     */
    function selectProvider(providerType) {
        selectedProvider = providerType;

        /* Toggle selected state on cards */
        var cards = document.querySelectorAll('.provider-card');
        for (var i = 0; i < cards.length; i++) {
            var card = cards[i];
            if (card.getAttribute('data-provider') === providerType) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        }

        _updateStep2NextButton();
    }

    /**
     * Send a connection test request via HTMX / fetch.
     * @param {string} providerType - The provider to test.
     */
    function testConnection(providerType) {
        var payload = _gatherProviderCredentials(providerType);
        if (!payload) return;

        var resultEl = document.getElementById('connection-result-' + providerType);
        var spinnerEl = document.getElementById('test-spinner-' + providerType);
        if (spinnerEl) spinnerEl.classList.add('htmx-request');

        /* Use fetch to POST and get the HTMX partial back */
        var formData = new FormData();
        formData.append('provider_type', providerType);
        for (var key in payload) {
            if (payload.hasOwnProperty(key)) {
                formData.append(key, payload[key]);
            }
        }

        fetch('/setup/test-connection', {
            method: 'POST',
            body: formData,
            headers: { 'HX-Request': 'true' }
        })
        .then(function (response) { return response.text(); })
        .then(function (html) {
            if (resultEl) resultEl.innerHTML = html;
            /* Check if connection succeeded */
            if (html.indexOf('connection-success') !== -1) {
                connectionTested[providerType] = true;
            }
            _updateStep2NextButton();
        })
        .catch(function (err) {
            if (resultEl) {
                resultEl.innerHTML =
                    '<div class="connection-error">' +
                    '<span class="status-icon status-error">&#10007;</span>' +
                    '<span class="status-text">Network error: could not reach server.</span>' +
                    '</div>';
            }
        })
        .finally(function () {
            if (spinnerEl) spinnerEl.classList.remove('htmx-request');
        });
    }

    /**
     * Gather credentials from the form fields for a given provider.
     * @param {string} providerType
     * @returns {Object|null} Key-value credential data.
     */
    function _gatherProviderCredentials(providerType) {
        var data = {};
        if (providerType === 'ollama') {
            data.base_url = document.getElementById('ollama-url').value;
            data.model = document.getElementById('ollama-model').value;
        } else if (providerType === 'anthropic') {
            data.api_key = document.getElementById('anthropic-key').value;
            data.model = document.getElementById('anthropic-model').value;
        } else if (providerType === 'openai') {
            data.api_key = document.getElementById('openai-key').value;
            data.model = document.getElementById('openai-model').value;
        } else if (providerType === 'openai_compatible') {
            data.base_url = document.getElementById('compat-url').value;
            data.api_key = document.getElementById('compat-key').value;
            data.model = document.getElementById('compat-model').value;
        }
        return data;
    }

    /** Enable/disable the Step 2 Next button based on selection + test. */
    function _updateStep2NextButton() {
        var btn = document.getElementById('step2-next');
        if (!btn) return;
        btn.disabled = !(selectedProvider && connectionTested[selectedProvider]);
    }

    /** Save the selected provider config via POST, then advance. */
    function saveProvider() {
        if (!selectedProvider) return;
        var payload = _gatherProviderCredentials(selectedProvider);
        if (!payload) return;

        var formData = new FormData();
        formData.append('provider_type', selectedProvider);
        for (var key in payload) {
            if (payload.hasOwnProperty(key)) {
                formData.append(key, payload[key]);
            }
        }

        fetch('/setup/save-provider', {
            method: 'POST',
            body: formData,
            headers: { 'HX-Request': 'true' }
        })
        .then(function (response) {
            if (response.ok) {
                next();
            }
        });
    }

    /* ------------------------------------------------------------------ */
    /* Search provider selection (Step 3)                                  */
    /* ------------------------------------------------------------------ */

    /**
     * Select a search provider card.
     * @param {string} searchType - One of: tavily, brave, serpapi
     */
    function selectSearch(searchType) {
        selectedSearch = searchType;

        var cards = document.querySelectorAll('.search-card');
        for (var i = 0; i < cards.length; i++) {
            var card = cards[i];
            if (card.getAttribute('data-search') === searchType) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        }

        var btn = document.getElementById('step3-next');
        if (btn) btn.disabled = false;
    }

    /** Save the search config and advance. */
    function saveSearch() {
        if (!selectedSearch) return;

        var keyEl = document.getElementById(selectedSearch + '-key');
        var apiKey = keyEl ? keyEl.value : '';

        var formData = new FormData();
        formData.append('provider', selectedSearch);
        formData.append('api_key', apiKey);

        fetch('/setup/save-search', {
            method: 'POST',
            body: formData,
            headers: { 'HX-Request': 'true' }
        })
        .then(function (response) {
            if (response.ok) {
                next();
            }
        });
    }

    /** Skip search setup — store provider as 'none'. */
    function skipSearch() {
        var formData = new FormData();
        formData.append('provider', 'none');
        formData.append('api_key', '');

        fetch('/setup/save-search', {
            method: 'POST',
            body: formData,
            headers: { 'HX-Request': 'true' }
        })
        .then(function () {
            next();
        });
    }

    /* ------------------------------------------------------------------ */
    /* Tour carousel (Step 4)                                              */
    /* ------------------------------------------------------------------ */

    /** Show the next tour slide. */
    function tourNext() {
        if (tourSlide < totalTourSlides - 1) {
            _showTourSlide(tourSlide + 1);
        }
    }

    /** Show the previous tour slide. */
    function tourPrev() {
        if (tourSlide > 0) {
            _showTourSlide(tourSlide - 1);
        }
    }

    /**
     * Display a specific tour slide.
     * @param {number} index - Slide index (0-based).
     */
    function _showTourSlide(index) {
        var slides = document.querySelectorAll('.tour-slide');
        for (var i = 0; i < slides.length; i++) {
            slides[i].classList.remove('active');
        }
        var target = document.querySelector('.tour-slide[data-slide="' + index + '"]');
        if (target) target.classList.add('active');

        tourSlide = index;

        /* Update nav buttons */
        var prevBtn = document.getElementById('tour-prev');
        var nextBtn = document.getElementById('tour-next');
        var counter = document.getElementById('tour-counter');

        if (prevBtn) prevBtn.disabled = (index === 0);
        if (nextBtn) nextBtn.disabled = (index === totalTourSlides - 1);
        if (counter) counter.textContent = (index + 1) + ' / ' + totalTourSlides;
    }

    /**
     * Interactive quick tour highlighting UI elements in sequence.
     * This is an alternative to the carousel approach; it walks through
     * highlights with overlay explanations. Currently, the carousel
     * approach is used in the template.
     */
    function quickTour() {
        _showTourSlide(0);
    }

    /* ------------------------------------------------------------------ */
    /* Completion                                                          */
    /* ------------------------------------------------------------------ */

    /** Mark onboarding complete and redirect to home. */
    function complete() {
        fetch('/setup/complete', {
            method: 'POST',
            headers: { 'HX-Request': 'true' }
        })
        .then(function () {
            window.location.href = '/';
        });
    }

    /* ------------------------------------------------------------------ */
    /* Initialisation                                                       */
    /* ------------------------------------------------------------------ */

    /* On page load, detect Ollama status */
    document.addEventListener('DOMContentLoaded', function () {
        _detectOllama();
    });

    /** Probe the Ollama detection endpoint and update the status badge. */
    function _detectOllama() {
        var badge = document.getElementById('ollama-status');
        fetch('/setup/test-connection', {
            method: 'POST',
            body: new URLSearchParams({ provider_type: 'ollama', base_url: 'http://localhost:11434', model: '' }),
            headers: { 'HX-Request': 'true', 'Content-Type': 'application/x-www-form-urlencoded' }
        })
        .then(function (response) { return response.text(); })
        .then(function (html) {
            if (html.indexOf('connection-success') !== -1) {
                if (badge) {
                    badge.textContent = 'Running';
                    badge.classList.add('badge-success');
                }
            } else {
                if (badge) {
                    badge.textContent = 'Not detected';
                    badge.classList.add('badge-warning');
                }
            }
        })
        .catch(function () {
            if (badge) {
                badge.textContent = 'Not detected';
                badge.classList.add('badge-warning');
            }
        });
    }

    /* ------------------------------------------------------------------ */
    /* Public API                                                          */
    /* ------------------------------------------------------------------ */

    return {
        goToStep: goToStep,
        next: next,
        back: back,
        selectProvider: selectProvider,
        testConnection: testConnection,
        saveProvider: saveProvider,
        selectSearch: selectSearch,
        saveSearch: saveSearch,
        skipSearch: skipSearch,
        tourNext: tourNext,
        tourPrev: tourPrev,
        quickTour: quickTour,
        complete: complete
    };
})();
