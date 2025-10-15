# Directorio de exalumnos

Este proyecto implementa la primera iteración del directorio privado para tu curso. Parte desde una base en Django con:

- Modelo de usuario personalizado basado en correo electrónico.
- Perfil extendido con los campos acordados (familia, trabajo, consentimiento).
- Auditoría básica que guarda cada cambio relevante junto con la persona que lo realizó.
- Formulario web para que cada exalumno actualice únicamente su información.

## Requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/) o `pip`

## Primeros pasos (local)

1. **Clona el repositorio** en tu máquina de desarrollo:
   ```bash
   git clone <URL_DEL_REPO>
   cd pruebas_ai
   ```
2. **Crea y activa un entorno virtual** para aislar dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows usa .venv\Scripts\activate
   ```
3. **Instala las dependencias** definidas en `requirements.txt`:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Configura las variables de entorno** copiando el archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```
   Ajusta los valores según tu entorno (ver tabla más abajo).
5. **Aplica las migraciones de la base de datos** para crear las tablas iniciales:
   ```bash
   python manage.py migrate
   ```
6. **Crea tu usuario administrador** (usa el correo con el que gestionarás las invitaciones):
   ```bash
   python manage.py createsuperuser --email tu-correo@dominio.cl
   ```
7. **Inicia el servidor de desarrollo** y verifica que la aplicación responde:
   ```bash
   python manage.py runserver
   ```

Accede a `http://127.0.0.1:8000/` para entrar a la página de inicio y a `http://127.0.0.1:8000/admin/` para crear usuarios o revisar perfiles.

> Si al instalar paquetes encuentras bloqueos de red (por ejemplo, en un servidor sin acceso a PyPI), descarga previamente las ruedas de Django y cópialas a tu entorno interno. También puedes montar un mirror interno de PyPI.

## Flujo sugerido para invitar compañeros

1. **Crear usuarios desde el admin**: Entra a `/admin/accounts/user/add/` y registra el correo y nombre.
2. Marca "Enviar enlace de restablecer contraseña" manualmente desde el admin (menú acciones) o envía tú mismo una contraseña temporal.
3. Comparte la URL pública `https://tu-dominio/perfil/` para que completen sus datos.
4. Cada vez que actualicen información clave, se generará una entrada en el historial visible solo por el dueño del perfil y los administradores.

## Próximos pasos recomendados

- Implementar un flujo de invitaciones con token y auto-registro pendiente de aprobación.
- Añadir exportación a CSV/PDF desde el panel administrativo.
- Configurar almacenamiento PostgreSQL y HTTPS detrás de Nginx para el despliegue definitivo.
- Automatizar respaldos y rotación de logs de auditoría.
- Revisar la política de retención de datos y mecanismo de baja de usuarios.

## Variables de entorno relevantes

| Variable | Descripción |
|----------|-------------|
| `DJANGO_SECRET_KEY` | Clave secreta para producción. |
| `DJANGO_DEBUG` | `false` en producción. |
| `DJANGO_ALLOWED_HOSTS` | Lista separada por comas de dominios permitidos. |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Lista separada por comas con los orígenes HTTPS confiables. |

## Exportar datos

Desde el panel admin puedes usar acciones sobre `AlumniProfile` para exportar CSV utilizando la acción integrada "Export selected profiles" que añadiremos en una iteración futura. Mientras tanto, Django permite exportar desde la vista de cambio en formato JSON mediante la acción "Export as JSON" si instalas `django-import-export`.

## Testing

Ejecuta la suite básica (a completar en iteraciones posteriores):

```bash
python manage.py test
```

Actualmente no hay tests automatizados definidos; se recomienda agregarlos cuando empecemos a trabajar los flujos de invitaciones y baja de usuarios.
