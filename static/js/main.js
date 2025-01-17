document.addEventListener("DOMContentLoaded", function() {
    const recommendationContainer = document.getElementById("recommendation");

    if (recommendationContainer) {
        recommendationContainer.classList.add("show");
        redirection(); // Llama a la función redirección
        
    }

    
});
function redirection(){
    const tiempoDeEspera = 10000; // 10 segundos

    setTimeout(() => {
        // Cambia a la ruta del home
        window.location.href = '/'; // Asegúrate de que '/' sea la ruta del home
    }, tiempoDeEspera);

}



